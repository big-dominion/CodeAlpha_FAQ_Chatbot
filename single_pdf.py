"""
DOMINION LexOracle: Single-PDF Quick Verification
-------------------------------------------------

WHAT THIS FILE DOES (high-level summary):
This is the simplest, most ad-hoc diagnostic script in the project -
smaller and more disposable than `measure_data.py` or
`verify_all_sources.py`. It exists purely as a fast manual sanity check on
ONE specific PDF at a time: "does pypdf's text extraction actually produce
readable, sensible text for this file at all?" It is meant to be run and
edited by hand whenever a new source PDF is added, or whenever an existing
one is behaving oddly during ingestion (e.g. producing very few chunks, or
chunks that look garbled) - a quick first check before reaching for the
heavier diagnostics in `measure_data.py` (which checks section-boundary
regex matching) or `verify_all_sources.py` (which runs the full robust
chunker and reports on every document in the docs folder at once).

It is hardcoded to check a single file, "acja_2015.pdf" - to check a
different PDF, the filename string on the second line needs to be edited
directly. There is no command-line argument handling, no function
definitions, and no `if __name__ == "__main__":` guard at all here - the
whole file is just four lines of top-level script that run immediately
whenever this file is executed.

What it actually checks, step by step:
1. Opens the named PDF with pypdf's `PdfReader`.
2. Extracts the text of just the FIRST page (`reader.pages[0]`) and prints
   only its first 800 characters - enough to eyeball whether extraction
   produced normal, readable prose (a good sign) versus garbled/empty/
   mostly-whitespace text (a sign this PDF may need special handling, or
   that its text layer is missing entirely, e.g. because it's a scanned
   image rather than real embedded text).
3. Prints a visual separator line.
4. Prints the total number of pages in the PDF, as a quick check that the
   file was recognized correctly and has a page count that looks right
   for the document it's supposed to be (e.g. catching a corrupted or
   truncated download that opens but reports an unexpectedly tiny page
   count).

Nothing here writes to Pinecone, touches the embedding pipeline, or
imports anything from services/ - it is entirely local, read-only, and
has no dependency on any of the app's environment variables (unlike every
other script in this project, it doesn't even call load_dotenv(), since
it needs no API keys at all for this check).
"""
from pypdf import PdfReader

# Opens the specific PDF being checked. To verify a different document,
# edit this filename directly - this script is meant to be hand-edited
# per use, not parameterized via a CLI argument.
reader = PdfReader("./data/docs/acja_2015.pdf")

# Prints just the first 800 characters of the FIRST page's extracted text,
# as a quick eyeball check that pypdf is pulling out real, readable
# statute text from this PDF (rather than empty text, garbled characters,
# or OCR artifacts) before trusting it to a full ingestion run.
print(reader.pages[0].extract_text()[:800])
print("---")

# Prints the total page count, as a sanity check that the file opened
# correctly and has roughly the number of pages expected for this
# document - useful for catching a truncated or otherwise corrupted PDF
# that still opens without error but is missing most of its content.
print(f"Total pages: {len(reader.pages)}")