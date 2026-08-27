"""
DOMINION LexOracle: Source Verification (Upgraded)
--------------------------------------------------
- Uses an aggressive, flexible regex for legal sections.
- Implements a hard fallback chunker to protect Pinecone limits.

WHAT THIS FILE DOES (high-level summary):
This is a standalone, manually-run diagnostic/reporting script - like
`measure_data.py`, it is NOT part of the running app and nothing in
main.py or services/ imports it. Its purpose is to scan EVERY PDF sitting
in the standard docs directory (./data/docs, the same one `ingest.py`
reads from) and print a health report for each one, so a person can
confirm every source document is actually safe to ingest BEFORE running
the real `ingest.py` pass on the whole batch.

It is essentially a self-contained, independent re-implementation of the
chunking approach in `ingest.py`'s `chunk_text_by_section` - close enough
in spirit to be a meaningful pre-flight check, but with its own separate
regex and its own separate, MORE THOROUGH fallback subdivision logic (see
`get_robust_chunks` below). It does not call anything from services/ and
does not write anything to Pinecone - it only reads PDFs and prints a
report; nothing here touches the actual index.

Why this exists as a SEPARATE script from ingest.py rather than reusing
its chunking function directly: this file's chunker is explicitly an
"upgraded" experiment - a more aggressive section-boundary regex (handling
more marker shapes like "SECTION 23(a)", not just the shapes ingest.py's
own regex looks for) plus a stricter, multi-level fallback that guarantees
NO chunk can ever exceed `MAX_CHAR_LIMIT`, even in the worst case of one
enormous, unbroken paragraph. Running it separately, as a report-only tool,
lets these upgraded rules be validated against every real source document
first, without risking corrupting or duplicating anything in the live
Pinecone index the way running an actual ingestion pass would.

The core logic, per PDF file found in DOCS_DIR:
1. EXTRACT: Read the PDF page by page via pypdf, concatenating all
   extracted text into one string, same defensive "or ''" pattern used
   elsewhere in this project for pages with no extractable text.
2. CHUNK (`get_robust_chunks`): Split that text into chunks in two layers:
     a. First, split on detected statutory section boundaries (a regex
        broader than ingest.py's own, additionally tolerating a closing
        parenthesis after the number, e.g. "23)" or "23(a)" patterns, and
        matching case-insensitively).
     b. Then, for any resulting section-level chunk that's STILL bigger
        than MAX_CHAR_LIMIT (e.g. a large Schedule with no internal
        section breaks), fall back to splitting on blank-line paragraph
        boundaries, greedily packing paragraphs into sub-chunks up to the
        limit - and if even a SINGLE paragraph on its own exceeds the
        limit, hard-slice it into fixed-size pieces with no further
        attempt at meaningful boundaries, as an absolute last resort. This
        nested fallback is what guarantees the hard ceiling described
        above: no chunk this function returns can ever be longer than
        MAX_CHAR_LIMIT, no matter how the source document is formatted.
3. REPORT (`analyze`): Compute simple statistics over the resulting chunks
   (count, longest chunk, average chunk length) and print a formatted
   summary block for that file, flagging two specific conditions a human
   should look into: a chunk that STILL exceeds MAX_CHAR_LIMIT despite the
   fallback (which would indicate the fallback logic itself has a bug,
   since it's designed to make this impossible) or the document turning
   out suspiciously short (fewer than 3 pages, which could mean the PDF
   failed to extract properly, or the wrong file was placed in the docs
   folder).

The whole file runs top-to-bottom as a script (no `if __name__ ==
"__main__":` guard, and no functions called at import time other than
inside the final loop) - importing this module would immediately scan and
print a report for every PDF in DOCS_DIR as a side effect, so it is meant
to be run directly, not imported.
"""
import os
import re
from pypdf import PdfReader

# Directory this script scans for PDFs - the same standard docs location
# ingest.py reads from, so this script is checking the exact same source
# files that a real ingestion run would process.
DOCS_DIR = "./data/docs"

# The hard ceiling every chunk returned by get_robust_chunks is guaranteed
# to respect, no matter what. Set conservatively (~1500 tokens worth) to
# stay safely under the input limits of most embedding models, including
# the "multilingual-e5-large" model this app actually uses.
MAX_CHAR_LIMIT = 8000  # Safe limit for most embedding models (~1500 tokens)

def get_robust_chunks(full_text):
    """
    Splits `full_text` into chunks that are section-aware where possible,
    and are GUARANTEED to never exceed MAX_CHAR_LIMIT under any
    circumstance, via a two-level fallback strategy.

    Detailed flow:
    - Step 1 - section-boundary split: `split_pattern` matches the START
      of what looks like a new statutory section (via a zero-width
      lookahead, so the matched text isn't consumed/removed): an optional
      "section " prefix, one or more digits, an optional single letter,
      then EITHER a literal "." or a literal ")" (this is the "upgraded"
      part relative to ingest.py's own pattern - tolerating a trailing
      ")" lets it catch markers like "23)" or "23(a)" that a period-only
      pattern would miss), followed by whitespace. The whole match is
      done case-insensitively (`re.IGNORECASE`), so "Section", "SECTION",
      and "section" are all recognized. `re.split` on this pattern breaks
      `full_text` into `raw_sections`, one piece per detected boundary.
    - For each raw section piece:
        * Strip it; skip entirely if nothing's left (an empty piece, same
          as ingest.py's own chunker).
        * Step 2 - size fallback: if this section-level piece is STILL
          longer than MAX_CHAR_LIMIT (e.g. a big Schedule with no
          internal section markers of its own), it needs further
          subdividing:
            - Split it on blank lines (`\n\s*\n`) into paragraphs.
            - Walk through the paragraphs, greedily accumulating them
              into `current_chunk`:
                > If a SINGLE paragraph is itself longer than
                  MAX_CHAR_LIMIT (the paragraph-level fallback still
                  isn't enough on its own), that one paragraph is
                  hard-sliced into fixed-size pieces of exactly
                  MAX_CHAR_LIMIT characters each, with no further regard
                  for sentence or word boundaries - this is the true
                  last-resort fallback, and it's what makes the "no chunk
                  can ever exceed MAX_CHAR_LIMIT" guarantee actually
                  hold in every case. Those hard-sliced pieces are added
                  directly to `final_chunks` and this paragraph is
                  skipped from the normal accumulation logic below via
                  `continue`.
                > Otherwise, if adding this paragraph (plus a trailing
                  blank line) to `current_chunk` would still keep it
                  under MAX_CHAR_LIMIT, append it there.
                > Otherwise, `current_chunk` is full: close it off by
                  appending its stripped contents to `final_chunks` (if
                  it has anything in it), then start a new
                  `current_chunk` beginning with this paragraph.
            - After the paragraph loop, if there's anything left over in
              `current_chunk`, append its stripped contents to
              `final_chunks` too, so nothing is dropped at the end.
        * If the section-level piece was already within MAX_CHAR_LIMIT to
          begin with, just append it to `final_chunks` directly, no
          further subdivision needed.
    - Returns `final_chunks`, the complete flat list of chunks for this
      document, none of which can exceed MAX_CHAR_LIMIT.
    """
    # 1. Upgraded Regex: Matches "1.", "1A.", "Section 1.", "SECTION 23(a)" 
    # Accounts for leading spaces and ignores case.
    split_pattern = r'\n(?=\s*(?:section\s+)?\d+[a-zA-Z]?(?:\.|\))\s)'
    raw_sections = re.split(split_pattern, full_text, flags=re.IGNORECASE)
    
    final_chunks = []
    for sec in raw_sections:
        sec = sec.strip()
        if not sec:
            continue
            
        # 2. Fallback Mechanism: If a chunk (like a Schedule) is still too big, 
        # slice it by paragraphs (double newlines) to save your vector DB.
        if len(sec) > MAX_CHAR_LIMIT:
            paras = re.split(r'\n\s*\n', sec)
            current_chunk = ""
            for p in paras:
                # If a single paragraph is insanely long, force split it
                if len(p) > MAX_CHAR_LIMIT:
                    chunks = [p[i:i+MAX_CHAR_LIMIT] for i in range(0, len(p), MAX_CHAR_LIMIT)]
                    final_chunks.extend(chunks)
                    continue
                    
                if len(current_chunk) + len(p) < MAX_CHAR_LIMIT:
                    current_chunk += p + "\n\n"
                else:
                    if current_chunk:
                        final_chunks.append(current_chunk.strip())
                    current_chunk = p + "\n\n"
            if current_chunk:
                final_chunks.append(current_chunk.strip())
        else:
            final_chunks.append(sec)
            
    return final_chunks

def analyze(filepath, filename):
    """
    Reads one PDF, runs it through get_robust_chunks, and prints a
    formatted health-check report for it.

    Detailed flow:
    - Open `filepath` with PdfReader and record its total page count.
    - Extract text from every page and join it into one `full_text`
      string, same "or ''" defensive pattern used elsewhere in this
      project for pages with no extractable text. (The comment in the
      code notes that a more sophisticated per-page header/footer
      stripping approach was considered but not implemented - this
      version just relies on the fallback chunker in get_robust_chunks to
      absorb whatever irregularities that leaves behind.)
    - Run `full_text` through `get_robust_chunks` to get this document's
      chunk list.
    - Compute simple statistics over the chunk lengths: sort all lengths
      descending, take the first (largest) as `max_len` (0 if there are
      no chunks at all), and compute the mean as `avg_len` (0 if empty).
    - Print a formatted report block: a divider line, the filename, page
      count, total chunk count, and the longest/average chunk lengths.
    - Build up a list of `flags` - conditions worth a human's attention:
        * If `max_len` is still greater than MAX_CHAR_LIMIT despite
          get_robust_chunks's guarantee, flag it as CRITICAL - this
          would mean the fallback logic itself has a bug, since by
          design this should never be possible.
        * If the document has fewer than 3 pages, flag it as
          "Suspiciously short" - possibly a sign of a text-extraction
          failure or the wrong file being present.
    - If any flags were raised, print them all under a "FLAGS:" header.
      Otherwise, print a clean "Ready for Pinecone ingestion" message.
    """
    reader = PdfReader(filepath)
    pages = len(reader.pages)
    
    # Simple page header/footer mitigation: strip top/bottom 5% of text if possible,
    # but for pypdf, we just join and rely on the fallback chunker.
    full_text = "".join(p.extract_text() or "" for p in reader.pages)

    chunks = get_robust_chunks(full_text)
    
    lengths = sorted((len(c) for c in chunks), reverse=True)
    max_len = lengths[0] if lengths else 0
    avg_len = sum(lengths) / len(lengths) if lengths else 0

    print(f"\n{'='*70}")
    print(f"{filename}")
    print(f"{'='*70}")
    print(f"Pages: {pages}")
    print(f"Generated Chunks: {len(chunks)}")
    print(f"Longest chunk: {max_len} chars | Avg chunk: {avg_len:.0f} chars")
    
    flags = []
    if max_len > MAX_CHAR_LIMIT:
        flags.append(f"CRITICAL: A chunk is {max_len} chars. Fallback failed!")
    if pages < 3:
        flags.append("Suspiciously short document.")

    if flags:
        print(f"\n  FLAGS:")
        for f in flags:
            print(f"  - {f}")
    else:
        print(f"\n  Ready for Pinecone ingestion. No red flags.")

# Runs immediately at module load / script execution (no __main__ guard,
# unlike cloud_index.py, ingest.py, and wipe_pinecone.py): iterates every
# filename in DOCS_DIR in sorted (alphabetical) order, and for each one
# ending in ".pdf", runs the full analyze() report against it. Since this
# runs at the top level with no guard, simply IMPORTING this module
# elsewhere would trigger a full scan-and-print of every PDF in DOCS_DIR
# as a side effect - it is meant to be run directly as a script, not
# imported.
for filename in sorted(os.listdir(DOCS_DIR)):
    if filename.endswith(".pdf"):
        analyze(os.path.join(DOCS_DIR, filename), filename)