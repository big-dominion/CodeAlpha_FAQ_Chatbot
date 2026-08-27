"""
DOMINION LexOracle: Cloud-Accelerated PDF Ingestion Engine
----------------------------------------------------------

WHAT THIS FILE DOES (high-level summary):
This is a one-off/batch data-loading script (run manually, like
`cloud_index.py`, not called during normal request handling) that takes
raw statute PDFs sitting on disk and turns them into searchable vectors in
the Pinecone index the running app queries at request time
(`services/vector_store.py`). It is the ingestion counterpart to
`cloud_index.py` (which provisions the empty index) and `vector_store.py`
(which reads from it) - this file is what actually fills it with data.

The pipeline, per PDF file in `./data/docs`:
1. READ: Every page of the PDF is text-extracted via `pypdf` and
   concatenated into one big string for that Act.
2. CHUNK (`chunk_text_by_section`): That full text is split into smaller
   pieces along statutory section boundaries wherever possible (so a
   retrieved chunk corresponds to a coherent section of law rather than an
   arbitrary character cutoff), with any resulting chunk that's still too
   large further subdivided into fixed-size, overlapping pieces so nothing
   fed downstream is too big for the embedding model or search context.
3. LABEL (`extract_section_label`): Each chunk is inspected to pull out
   its section number/label for metadata purposes (e.g. "Section 152"),
   with specific logic to avoid misreading a 4-digit enactment year at the
   start of a chunk (e.g. "2020" from a Gazette header or an Act's title)
   as if it were itself a section number.
4. EMBED: Chunks are grouped into batches of up to 64 and sent to
   Pinecone's hosted "multilingual-e5-large" inference endpoint to be
   turned into 1024-dimension vectors - explicitly as "passage"-type
   input this time (as opposed to "query"-type input used when embedding
   a search question in vector_store.py), since the e5 model family
   embeds documents/passages differently from search queries by design.
5. UPSERT: Each embedded chunk, together with its metadata (act name,
   section label, its index within the Act, and its own raw text), is
   written into the Pinecone index in the same batches, under a
   predictable ID of the form "{ACT_NAME}-chunk-{chunk_index}".

Why chunking tries to respect section boundaries first, rather than just
splitting on a fixed character count throughout: a chunk that starts and
ends mid-section, with no awareness of where one statutory provision ends
and the next begins, would be a poor and confusing thing to hand to the
LLM as "the text of Section X" - retrieval and citation quality both
depend on chunks lining up with actual sections as closely as possible.

Why `INDEX_NAME` here reads from the environment with the exact same
"lexoracle-cloud" fallback as `services/vector_store.py`: if this script's
default index name ever drifted out of sync with the one the running app
actually queries, ingestion would appear to succeed with no errors, while
the live app would keep finding nothing for every question - a silent,
hard-to-diagnose failure mode. Keeping both fallbacks identical is a
deliberate guard against that.

This script is safe to re-run: each chunk's vector ID is deterministic
(derived from the act name and its position in that act's chunk list), so
re-ingesting the same PDF again will simply overwrite the same vector IDs
with fresh data rather than duplicating entries - though note this means a
change to `chunk_text_by_section`'s behavior between runs could shift
chunk boundaries/positions and therefore change which text ends up under
which ID.

Like `cloud_index.py`, this is guarded by `if __name__ == "__main__":` at
the bottom, so importing this module elsewhere does not automatically
trigger ingestion.
"""

import os
import re
from pypdf import PdfReader
from pinecone import Pinecone
from dotenv import load_dotenv

# Loads variables from a local .env file (e.g. PINECONE_API_KEY,
# PINECONE_INDEX_NAME) into the process environment before anything below
# tries to read them.
load_dotenv()

# The Pinecone client used for both embedding (pc.inference.embed) and
# writing vectors (index.upsert) below. Reads the API key from the
# environment - no placeholder fallback here, so a missing key will fail
# loudly the first time this script actually talks to Pinecone, which is
# acceptable for a manually-run ingestion script.
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

# Reads the index name from .env instead of hardcoding it, so ingestion
# always writes to the same index vector_store.py reads from. Falls back
# to "lexoracle-cloud" - the name this app has always used - if the env
# var is ever unset. Keeping this fallback identical to vector_store.py's
# matters: a mismatch between the two would mean ingestion silently
# writes to a DIFFERENT index than the one the running app queries,
# which would look like "ingestion succeeded but retrieval finds
# nothing" with no error anywhere to point at why.
INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "lexoracle-cloud")


def chunk_text_by_section(text: str, max_chunk_size: int = 4000, overlap: int = 200) -> list:
    """Splits statute text on section boundaries; subdivides oversized blocks.

    Detailed flow:
    - `split_pattern` matches the START of what looks like a new
      statutory section: a newline followed by an optional "SECTION "
      prefix, a number (with an optional trailing letter, e.g. "12A"),
      and then either ". " (a period and space, as in "12. Definitions")
      or a space followed by an uppercase letter (as in "12 Interpretation").
      The lookahead `(?=...)` means the split happens exactly AT that
      newline, without consuming/removing the matched section-start text
      itself, so each resulting piece still begins with its own section
      marker intact.
    - `re.split` with that pattern breaks the full document text into raw
      chunks along every detected section boundary.
    - For each raw chunk:
        * Strip surrounding whitespace; skip it entirely if that leaves
          nothing (an empty piece, e.g. from text before the very first
          detected section boundary).
        * If the chunk is still longer than `max_chunk_size` (a single
          section's text was simply too long on its own), subdivide it
          further into fixed-size pieces of `max_chunk_size` characters,
          stepping forward by `max_chunk_size - overlap` each time so
          consecutive pieces share `overlap` characters of context at
          their boundary - this overlap helps avoid losing meaning for
          content that happens to fall right at a hard cut point.
        * Otherwise (it already fits within the limit), keep the chunk
          as a single piece unchanged.
    - Returns the final flat list of chunks, ready for section-label
      extraction and embedding.
    """
    split_pattern = r'\n(?=\s*(?:SECTION\s+)?\d+[a-zA-Z]?(?:\.\s+|\s+[A-Z]))'
    raw_chunks = re.split(split_pattern, text)

    chunks = []
    for chunk in raw_chunks:
        chunk = chunk.strip()
        if not chunk:
            continue

        if len(chunk) > max_chunk_size:
            start = 0
            while start < len(chunk):
                end = start + max_chunk_size
                chunks.append(chunk[start:end])
                start += max_chunk_size - overlap
        else:
            chunks.append(chunk)

    return chunks


def extract_section_label(text: str) -> str:
    """Extracts valid section headers while suppressing statutory enactment years.

    Detailed flow:
    - Try to match the very start of `text` against the same general
      "section marker" shape used for splitting above (optional "SECTION "
      prefix, a number with an optional trailing letter, then ". " or a
      space + uppercase letter), this time actually capturing the
      number/letter portion as `label_num`.
    - If it matches:
        * Upper-case the captured label for consistency (e.g. "12a"
          becomes "12A").
        * If that label is one of a known list of 4-digit years
          ("2020", "2015", "1999", "2011", "2023") - years this app's
          source PDFs are known to have appearing at the start of a
          Gazette header or an Act's own title line, which would
          otherwise get misread as if "Section 2020" existed - return
          "General Provision" instead of treating it as a real section
          number.
        * Otherwise, return it formatted as "Section {label_num}".
    - If the text doesn't match the section-marker pattern at all (e.g.
      it's introductory text, a preamble, or anything not starting with a
      recognizable section header), return "General Provision" as a
      catch-all label.
    """
    match = re.match(r'^\s*(?:SECTION\s+)?(\d+[a-zA-Z]?)(?:\.\s+|\s+[A-Z])', text, flags=re.IGNORECASE)
    if match:
        label_num = match.group(1).upper()
        # Suppress false positives generated by Gazette headers or Act titles
        if label_num in ["2020", "2015", "1999", "2011", "2023"]:
            return "General Provision"
        return f"Section {label_num}"
    return "General Provision"


def ingest_pdfs():
    """
    The main ingestion routine: reads every PDF in the docs directory,
    chunks and labels its text, embeds those chunks remotely, and upserts
    the resulting vectors into Pinecone.

    Detailed flow:
    - Resolve the source PDF directory ("./data/docs") and get a handle
      to the target Pinecone index.
    - Iterate over every filename in that directory in sorted (alphabetical)
      order, processing only files ending in ".pdf":
        * Derive `act_name` from the filename itself (strip the ".pdf"
          extension, upper-case the rest) - this is both the display name
          stored in each chunk's metadata and the prefix used to build
          that act's vector IDs.
        * Open the PDF with `PdfReader` and extract text page by page,
          concatenating every page's extracted text (with a newline
          between pages) into one `full_text` string for the whole Act.
          A page that returns no extractable text (e.g. a scanned image
          page) is simply skipped rather than inserting an empty string.
        * Pass `full_text` through `chunk_text_by_section` to get the
          list of section-aware chunks for this Act.
        * Process those chunks in batches of 64 (Pinecone's inference
          endpoint is called once per batch rather than once per chunk,
          for efficiency):
            - Call `pc.inference.embed` with the batch of raw chunk
              strings, using the same "multilingual-e5-large" model as
              query-time embedding, but with `input_type: "passage"`
              instead of "query" - reflecting that these are documents
              being indexed, not a search question being asked.
            - For each chunk in the batch (paired with its corresponding
              embedding result via `zip`):
                > Compute its absolute position `chunk_index` within the
                  whole Act (not just within this batch), by adding the
                  batch's starting offset `i` to its position `j` inside
                  the batch.
                > Build its deterministic vector ID:
                  "{act_name}-chunk-{chunk_index}".
                > Extract its section label via `extract_section_label`.
                > Assemble its metadata dict: act name, section label,
                  its chunk index, and its own full raw text (this is
                  what lets vector_store.py's search results carry the
                  actual source text back to the RAG orchestrator without
                  a separate lookup).
                > Append the (vector_id, embedding_values, metadata)
                  tuple to this batch's `vectors` list.
            - Upsert the whole batch's vectors into the Pinecone index in
              one call, and print a progress line showing which batch
              number (out of the total for this Act) just completed.
    - After every PDF and every batch has been processed, print a final
      completion message.
    """
    docs_dir = "./data/docs"
    index = pc.Index(INDEX_NAME)

    for filename in sorted(os.listdir(docs_dir)):
        if filename.endswith(".pdf"):
            filepath = os.path.join(docs_dir, filename)
            act_name = filename.replace(".pdf", "").upper()
            print(f"\nProcessing {act_name}...")

            reader = PdfReader(filepath)
            full_text = ""
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    full_text += extracted + "\n"

            chunks = chunk_text_by_section(full_text)
            print(f"Generated {len(chunks)} chunks for {act_name}. Dispatching to Cloud GPU...")

            batch_size = 64
            total_batches = (len(chunks) + batch_size - 1) // batch_size

            for i in range(0, len(chunks), batch_size):
                batch_chunks = chunks[i:i + batch_size]

                # Generate 1024-dim dense vectors remotely via Pinecone GPU Inference
                embed_res = pc.inference.embed(
                    model="multilingual-e5-large",
                    inputs=batch_chunks,
                    parameters={"input_type": "passage"}
                )

                vectors = []
                for j, (chunk, emb_obj) in enumerate(zip(batch_chunks, embed_res)):
                    chunk_index = i + j
                    vector_id = f"{act_name}-chunk-{chunk_index}"
                    section_label = extract_section_label(chunk)

                    metadata = {
                        "act_name": act_name,
                        "section": section_label,
                        "chunk_index": chunk_index,
                        "text": chunk
                    }
                    vectors.append((vector_id, emb_obj.values, metadata))

                current_batch_num = (i // batch_size) + 1
                index.upsert(vectors=vectors)
                print(f"Uploaded batch {current_batch_num} / {total_batches} for {act_name}")

    print("\nPDF Cloud Ingestion complete!")


# Only runs the full ingestion pass when this file is executed directly
# (e.g. `python ingest.py`), not when it's imported as a module elsewhere
# in the app - so simply importing this file never has the side effect of
# reading PDFs or writing to Pinecone.
if __name__ == "__main__":
    ingest_pdfs()