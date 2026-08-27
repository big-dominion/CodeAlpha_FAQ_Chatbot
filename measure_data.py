"""
DOMINION LexOracle: Section-Boundary Diagnostic Script
------------------------------------------------------

WHAT THIS FILE DOES (high-level summary):
This is a standalone, manually-run diagnostic/debugging script - it is NOT
part of the running app and nothing in main.py or services/ imports it. Its
purpose is to sanity-check, for a given source PDF, how well the section-
splitting regex used in `ingest.py` (`chunk_text_by_section`'s
`split_pattern`) actually matches that specific document's real section
markers, BEFORE running a full ingestion pass on it.

Why this exists: `chunk_text_by_section` in ingest.py relies on a regex
pattern to detect where one statutory section ends and the next begins.
Different Acts' source PDFs can format their section numbers slightly
differently (e.g. "12." vs "12 " vs inconsistent spacing/OCR artifacts from
PDF text extraction), which means a pattern tuned against one Act's PDF
might silently miss boundaries in another. Missing a boundary means two (or
more) sections get merged into one oversized chunk instead of being split
correctly - a problem that's easy to miss just by skimming ingestion output,
since ingestion doesn't error out when this happens, it just produces worse
chunks. This script exists to catch that BEFORE ingestion, by directly
counting how many boundaries a simplified version of the real pattern finds
in a given PDF's extracted text, and by printing out lines that LOOK like
they could be a section marker but weren't actually matched - so a human can
visually eyeball whether those "possible missed marker" lines really are
missed section headers (meaning the ingestion regex needs adjusting for this
document) or just false alarms (e.g. an ordinary sentence or a page number
that happens to start with a digit).

How it works, per PDF filename passed to `diagnose`:
1. Open the PDF from the app's standard `./data/docs/` directory and
   extract text from every page, concatenating it all into one string (a
   page with no extractable text contributes an empty string rather than
   erroring, same defensive pattern used in ingest.py).
2. Run a regex (`\n(\d+[A-Z]?\.)`) over the WHOLE text to count how many
   times a newline is immediately followed by a number (with an optional
   trailing capital letter) and a literal period - a simplified stand-in
   for what a real section marker looks like in the source text - and
   print that count.
3. Split the same text into individual lines and, for roughly the first
   400 lines, look specifically for lines that:
     - Do NOT match the "number + optional letter + period" pattern at
       their very start (i.e. they weren't counted in step 2's match), but
     - DO match a looser pattern of "number + optional letter + a plain
       whitespace character" at their start (i.e. they still look like
       they COULD be a section number, just without the period after it,
       or with different punctuation/spacing).
   Any line matching that combination gets printed as a "possible missed
   marker" - a candidate the ingestion regex might be failing to catch for
   this particular PDF, worth a human's attention.

This is a debugging aid, not something with a return value or exit code
meant to be checked programmatically - its whole output is the printed
text meant to be read by a person deciding whether ingest.py's chunking
regex needs adjusting for a specific Act's PDF. It is invoked directly at
the bottom against two specific documents ("cama_2020.pdf" and
"acja_2015.pdf") as a hardcoded, ad-hoc check - to check a different or
additional file, add another `diagnose("filename.pdf")` call.
"""
import os, re
from pypdf import PdfReader

def diagnose(filename):
    """
    Runs the section-boundary diagnostic described in the module summary
    above against a single PDF file living in ./data/docs/.

    Detailed flow:
    - Open `filename` from the standard docs directory using PdfReader
      (the same library ingest.py uses for real ingestion).
    - Extract text from every page and join it all into one string
      `text`, with `or ""` guarding against a page that returns None for
      its extracted text (e.g. a scanned/image-only page) so that
      doesn't break the join.
    - Run `re.finditer` with the pattern `\n(\d+[A-Z]?\.)` over the whole
      `text`: this looks for a newline immediately followed by one or
      more digits, an optional single uppercase letter, and a literal
      period - e.g. matching "\n12." or "\n45A." - and collects every
      such match found anywhere in the document.
    - Print the filename alongside how many such matches were found, as
      a quick top-line number for how many section boundaries this
      simplified pattern detects in this specific PDF.
    - Split the full text into individual lines on newlines.
    - Loop over roughly the first 400 lines (`lines[:400]` - a sampling
      limit, not the whole document, since this is meant to be a quick
      manual spot-check rather than an exhaustive report) and for each
      line:
        * Strip it and check it against TWO patterns:
            1. `^\d+[A-Z]?\.` - does it start with digits, an optional
               capital letter, and a period? (this mirrors the stricter
               pattern used in the finditer count above)
            2. `^\d+[A-Z]?\s` - does it start with digits, an optional
               capital letter, and a plain whitespace character instead?
        * If the line does NOT match pattern 1 but DOES match pattern 2 -
          meaning it looks like it starts with what could be a section
          number, but not in the exact "digits + period" shape the main
          regex is looking for - print it (truncated to its first 70
          characters, with its raw repr so any hidden characters are
          visible) as a "possible missed marker" for a human to review.
    - The function has no return value; all of its output is these
      printed lines, meant to be read directly in the terminal.
    """
    reader = PdfReader(f"./data/docs/{filename}")
    text = "".join(p.extract_text() or "" for p in reader.pages)
    matches = list(re.finditer(r'\n(\d+[A-Z]?\.)', text))
    print(f"{filename}: {len(matches)} raw regex matches")
    # show a sample of what a plausible MISSED boundary looks like:
    # find lines that start with digits but weren't matched
    lines = text.split('\n')
    for i, line in enumerate(lines[:400]):
        if re.match(r'^\d+[A-Z]?\.', line.strip()) is None and re.match(r'^\d+[A-Z]?\s', line.strip()):
            print(f"  possible missed marker: {line.strip()[:70]!r}")

# Ad-hoc, hardcoded invocation against two specific Acts' PDFs - this is a
# manually-edited debugging script, not a general-purpose CLI tool, so
# checking a different or additional file means editing this call site
# directly (e.g. adding another diagnose("some_other_act.pdf") line)
# rather than passing a filename as an argument.
diagnose("cama_2020.pdf")
diagnose("acja_2015.pdf")