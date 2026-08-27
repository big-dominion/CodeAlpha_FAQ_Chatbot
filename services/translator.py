"""
DOMINION LexOracle: Translation Service
---------------------------------------

WHAT THIS FILE DOES:
Translates the AI's English answer into whichever language the user picked
(Yoruba, Hausa, Igbo, French, etc.). Public entry point: translate_text().

TWO PRIOR VERSIONS, AND WHY NEITHER WAS RIGHT ALONE:
  - FAST: one Google Translate call on the whole raw Markdown answer. Quick,
    but Google has no obligation to preserve "|" counts or "#" prefixes
    when translating a big blob of Markdown - it would often return a
    "successful" (non-empty, non-error-page) result that had nonetheless
    dropped a pipe from a table row or reworded a header prefix. Nothing in
    that version actually checked for THAT failure mode, so broken tables
    shipped silently, un-detected, most of the time in Yoruba/Hausa/Igbo.
  - CORRECT: every line routed through structure-aware handling (table
    rows cell-by-cell, headers with their "#" split off) and translated
    piece-by-piece, Google-then-MyMemory per piece. Never breaks structure,
    but pays for it with one network round-trip per piece - a table with
    20 cells means ~20 sequential requests before the answer is ready.

THIS VERSION: try the fast whole-answer Google call first, but don't trust
it blindly - validate that its Markdown "skeleton" (line count, per-row
pipe count, per-header "#" count) still matches the original before
accepting it. If it does, that's the fast path, done in one request. If it
doesn't - or Google fails outright - fall back to the slower, guaranteed-
correct per-piece method, but ONLY for that one request instead of every
request. See `_structure_preserved` for the exact check, and
`_translate_text_uncached` for where the two paths are stitched together.

Everything else - bold-stripping, typographic normalization, MyMemory's
garbage-result detection, the whole-answer-reverts-to-English-on-any-piece-
failure guarantee, and the translation cache - is unchanged from before.
"""
import logging
import os
import re
import time
from deep_translator import GoogleTranslator, MyMemoryTranslator

logger = logging.getLogger(__name__)

# MyMemory's free tier is far more restrictive for anonymous requests
# (roughly 5,000 words/day per IP) than for requests that identify a
# real email address (roughly 50,000 words/day) - registering an email
# costs nothing on MyMemory's side, it just tells them the traffic isn't
# anonymous scraping. Reading it from the environment, rather than
# hardcoding it here, keeps a real email address out of source control -
# set MYMEMORY_EMAIL in .env. If it's unset, every MyMemoryTranslator
# call below simply omits the email argument and falls back to the
# stricter anonymous tier, so this is safe to leave unset too.
MYMEMORY_EMAIL = os.environ.get("MYMEMORY_EMAIL")

# MyMemory's free tier rejects requests longer than roughly 500 characters.
# 450 leaves some room below that hard limit, so a line that is close to
# the edge does not get rejected by a small miscount.
MYMEMORY_CHUNK_LIMIT = 450

# Caches a completed translation by the exact (english_text, target_lang)
# pair that produced it. This exists specifically because the same
# English answer gets re-translated repeatedly during testing - asking
# the same question in five languages, switching the language dropdown
# back and forth on an answer already on screen, retrying after a
# failure, etc. Every one of those re-translates the identical text
# through the identical service for no new benefit, while still counting
# against Google's and MyMemory's rate limits. A cache hit costs nothing
# and never expires mid-process, since the English source for a given
# answer never changes.
#
# This is a plain in-memory dict, not shared across separate server
# processes and cleared on every restart - fine for a single-worker dev
# server. If this app is ever run with multiple worker processes or
# needs to survive restarts, this would need to move to something shared
# like Redis instead.
_translation_cache = {}


def _cache_key(text: str, target_lang: str) -> tuple:
    return (text, target_lang)

# MyMemory's free tier separately caps requests at roughly 5 per second
# (distinct from its much larger daily word quota). Translating a
# structured answer header-by-header and table-cell-by-cell, when the
# slower fallback path is used, can easily fire off many MyMemory
# requests for a single answer in quick succession - comfortably over
# that per-second cap if nothing paces them. _throttle_mymemory is called
# immediately before every MyMemory request and sleeps just long enough
# to keep consecutive calls at least this far apart.
_MYMEMORY_MIN_INTERVAL_SECONDS = 0.22
_last_mymemory_call_at = 0.0


def _throttle_mymemory():
    global _last_mymemory_call_at
    now = time.monotonic()
    wait = _MYMEMORY_MIN_INTERVAL_SECONDS - (now - _last_mymemory_call_at)
    if wait > 0:
        time.sleep(wait)
    _last_mymemory_call_at = time.monotonic()


# Substring MyMemory's own rate-limit error message contains, regardless
# of exact wording changes on their end. Used to tell "you're going too
# fast, try again shortly" apart from a genuine failure (bad language
# code, network drop, etc.) so only the former gets a retry.
_RATE_LIMIT_SIGNATURE = "too many requests"

lang_map = {
    "English": "en",
    "Yoruba": "yo",
    "Hausa": "ha",
    "Igbo": "ig",
    "French": "fr"
}

# MyMemory's API is stricter than Google's about language codes: it
# rejects a bare "yo" outright with "No support for the provided
# language", and only accepts the full locale-qualified form it lists in
# its own supported-languages response (e.g. "yo-NG", not "yo"). Google
# Translate has no such requirement and accepts the plain codes in
# lang_map above just fine - this second mapping exists ONLY for the
# MyMemory fallback path, so Google's request is never touched by it.
MYMEMORY_LANG_MAP = {
    "yo": "yo-NG",
    "ha": "ha-NE",
    "ig": "ig-NG",
    "fr": "fr-FR",
}

# GoogleTranslator scrapes translate.google.com instead of calling an
# official API. When Google rate-limits or blocks that scrape, it can
# hand back its own HTML error page's text instead of raising an
# exception - so there is no exception for a try/except to catch, and
# without this check that error page would be treated as a successful
# translation and shown to the user as if it were the real answer.
_GOOGLE_ERROR_SIGNATURES = ("Error 500", "Server Error", "That's an error")


def _looks_like_google_error_page(text: str) -> bool:
    if not text:
        return False
    return any(sig in text for sig in _GOOGLE_ERROR_SIGNATURES)


class _PieceTranslationFailed(Exception):
    """
    Raised when a single piece cannot be translated by either engine
    (Google failed or returned an error page, and MyMemory failed after
    its retry or returned a garbage result). Caught in
    _translate_text_uncached so the WHOLE answer falls back to English
    instead of mixing translated and untranslated pieces - a half-Hausa/
    half-English table looks broken regardless of why the one cell
    failed.
    """
    pass


# MyMemory occasionally returns a "successful" response (200 OK, no
# exception raised) whose body is not a real translation at all - just
# one character repeated many times. The regex flags a result that is
# 10+ consecutive repeats of a single character, tolerating up to 3
# stray characters before and/or after that run (a pure
# "0000000000000000" and a "Bsssssssssssssssssssss..." with a stray
# leading "B" have both been observed in practice - the looser tolerance
# catches both). _looks_like_garbage_result additionally requires that
# the SOURCE text isn't itself that same kind of repeated-character
# string, so a genuine input like "----------" is never wrongly flagged.
_GARBAGE_RESULT_RE = re.compile(r'^.{0,3}(.)\1{9,}.{0,3}$')


def _looks_like_garbage_result(source: str, result: str) -> bool:
    stripped_result = result.strip()
    stripped_source = source.strip()
    return bool(_GARBAGE_RESULT_RE.match(stripped_result)) and not _GARBAGE_RESULT_RE.match(stripped_source)


# A Markdown table's separator row (e.g. "|---|---|" or "|:--|--:|") is
# pure punctuation - there is nothing in it to translate, so the SLOW
# path passes it through completely untouched. The FAST path does NOT
# get that protection: it hands Google the whole answer as one blob,
# separator rows included, so Google is just as free to collapse a
# "|---|---|" (2 columns) down to "|---|" (1 column) as it is to mangle
# any other line. _structure_preserved below deliberately does NOT
# exempt separator rows from its pipe-count check for exactly that
# reason - a separator row whose column count no longer matches its
# table's header/data rows breaks the frontend's Markdown table parser
# just as badly as a data row losing a pipe does.
_TABLE_SEPARATOR_RE = re.compile(r'^\|?[\s\-\|\:]+\|?$')

# Matches a Markdown header line ("# ", "## ", up to "###### ") and
# captures the "#"*N + spacing prefix separately from the header text
# itself, so the prefix can be reattached untouched after only the text
# is translated (per-piece path), or checked for survival (fast-path
# validation).
_HEADER_RE = re.compile(r'^(#{1,6}\s+)(.*)$')

# Matches just the leading "#" run of a line, used by _structure_preserved
# to check a translated header line still starts with the SAME NUMBER of
# "#" characters as the original - independent of _HEADER_RE, which
# requires trailing text and won't match if Google mangled the spacing.
_LEADING_HASHES_RE = re.compile(r'^(#{1,6})(?:\s|$)')


def _structure_preserved(original: str, translated: str) -> bool:
    """
    Cheap post-hoc check used ONLY by the fast path: does a whole-answer
    Google translation still have the same Markdown "skeleton" as the
    input? A non-empty, non-error-page result from Google is NOT enough
    on its own to trust - Google can merge lines, drop a "|" from a table
    row, or reword a "#" header prefix while still returning what looks
    like a perfectly successful translation. This is exactly the failure
    mode that was slipping through silently before: nothing checked for
    it, so a translation that "succeeded" by every existing check could
    still render as a broken table on screen.

    Three things are checked, line by line:
      - Same number of lines overall. A merged or split line is already
        a sign the structure moved.
      - Every line containing "|" - a data row OR a separator row - has
        the SAME NUMBER of "|" characters in the translation as in the
        original. Separator rows are included deliberately: they're
        never translated on the slow path, but the fast path sends them
        to Google along with everything else, and a separator row whose
        column count silently drifted from its table's data rows breaks
        the frontend's table parser exactly as badly as a mangled data
        row does.
      - Every header line's leading "#" run survives with the exact same
        count in the translation.

    This intentionally does NOT try to validate that the translated
    CONTENT is correct or well-worded - only that the Markdown scaffolding
    around it is intact enough for the frontend to render it as the same
    kind of structure it started as. That's the only thing that was
    actually breaking on screen.
    """
    orig_lines = original.split("\n")
    trans_lines = translated.split("\n")

    if len(orig_lines) != len(trans_lines):
        return False

    for o_line, t_line in zip(orig_lines, trans_lines):
        o_stripped = o_line.strip()
        t_stripped = t_line.strip()

        if '|' in o_stripped:
            if o_line.count('|') != t_line.count('|'):
                return False

        header_match = _HEADER_RE.match(o_stripped)
        if header_match:
            orig_hashes = header_match.group(1).strip()
            trans_hashes_match = _LEADING_HASHES_RE.match(t_stripped)
            if not trans_hashes_match or trans_hashes_match.group(1) != orig_hashes:
                return False

    return True


def _strip_bold_markers(text: str) -> str:
    """
    Removes Markdown bold markers from the English answer BEFORE it is
    sent to be translated, and normalizes a leading "* " bullet marker to
    "- ".

    Machine translation does not reliably preserve a "**word**" pair
    intact - it can reorder the words inside the pair, or drop one of the
    two "**" markers entirely, leaving a stray, unpaired "*" or "**" in
    the output that has nothing left to pair with and never renders as
    real bold. Removing every asterisk before translation, in every
    position (not just inside a matched "**pair**"), means there is
    nothing left for the translation step to mangle. Bold styling is
    deliberately not preserved in translated languages as a result -
    structure (headings, bullets, tables) is unaffected, only bold
    emphasis is dropped.
    """
    stripped = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    stripped = re.sub(r'(^|\n)\*(\s+)', r'\1-\2', stripped)
    stripped = stripped.replace('*', '')
    return stripped


# The AI's raw_answer regularly contains "smart" typographic punctuation
# instead of the plain ASCII equivalent - curly quotes, a non-breaking
# hyphen (U+2011, NOT a regular "-"), the "§" section symbol, and an
# ellipsis character (U+2026) instead of three periods. These have been
# observed in practice to make MyMemory return degenerate garbage instead
# of a real translation for that piece of text.
_TYPOGRAPHIC_NORMALIZATION_MAP = {
    "\u2018": "'",   # left single quote
    "\u2019": "'",   # right single quote / apostrophe
    "\u201c": '"',   # left double quote
    "\u201d": '"',   # right double quote
    "\u2011": "-",   # non-breaking hyphen
    "\u2012": "-",   # figure dash
    "\u2013": "-",   # en dash
    "\u2014": "-",   # em dash
    "\u2015": "-",   # horizontal bar
    "\u2026": "...", # ellipsis
    "\u00a0": " ",   # non-breaking space
    "\u00a7": "Section ",  # section symbol "§"
}


def _normalize_typographic_chars(text: str) -> str:
    for char, replacement in _TYPOGRAPHIC_NORMALIZATION_MAP.items():
        text = text.replace(char, replacement)
    return text


def _split_long_line(line: str, max_len: int) -> list:
    """
    Breaks a single line into pieces no longer than max_len, cutting on
    sentence boundaries wherever possible. Only used on the MyMemory
    fallback path (Google has no comparable character limit).
    """
    sentences = line.replace("\n", " ").split(". ")
    chunks = []
    current = ""

    for sentence in sentences:
        piece = sentence if sentence.endswith(".") else sentence + "."
        if len(current) + len(piece) + 1 <= max_len:
            current = (current + " " + piece).strip()
        else:
            if current:
                chunks.append(current)
            current = piece if len(piece) <= max_len else piece[:max_len]

    if current:
        chunks.append(current)
    return chunks


def _translate_one_piece_mymemory(piece: str, lang_code: str) -> str:
    """
    Translates a single MyMemory-sized piece of text via MyMemory, in
    isolation. Only reached from _translate_run, and only when Google has
    already failed for that specific piece.

    Makes up to two attempts: a rate-limit response (matched against
    _RATE_LIMIT_SIGNATURE) gets one retry after a longer pause, since
    that failure mode is almost always transient; anything else raises
    _PieceTranslationFailed immediately rather than quietly returning
    English for just this one piece - see the module docstring for why a
    consistent single-language answer beats a partially-translated one.
    """
    mymemory_code = MYMEMORY_LANG_MAP.get(lang_code, lang_code)

    translator_kwargs = {"source": "en-US", "target": mymemory_code}
    if MYMEMORY_EMAIL:
        translator_kwargs["email"] = MYMEMORY_EMAIL

    for attempt in (1, 2):
        _throttle_mymemory()
        try:
            result = MyMemoryTranslator(**translator_kwargs).translate(piece)

            if not result:
                raise _PieceTranslationFailed(piece)

            if _looks_like_garbage_result(piece, result):
                logger.warning(
                    f"MyMemory returned a garbage result for one piece, "
                    f"failing the whole answer back to English. Piece: {piece[:60]!r}"
                )
                raise _PieceTranslationFailed(piece)

            return result
        except _PieceTranslationFailed:
            raise
        except Exception as e:
            if attempt == 1 and _RATE_LIMIT_SIGNATURE in str(e).lower():
                logger.warning(f"MyMemory rate-limited this piece, waiting then retrying once. Error: {e}")
                time.sleep(1.5)
                continue
            logger.warning(f"MyMemory failed on one piece, failing the whole answer back to English. Error: {e}")
            raise _PieceTranslationFailed(piece) from e

    raise _PieceTranslationFailed(piece)


def _translate_run(text: str, lang_code: str) -> str:
    """
    Translates one run of plain text - a whole plain line, a header's
    text (prefix already removed), or a single table cell's contents.
    Tries Google first (no chunking needed - Google has no ~500-char
    limit), falling back to MyMemory (which DOES need chunking) only if
    Google fails for this specific piece.

    This is the SLOW PATH's per-piece engine, only reached when the fast
    whole-answer attempt in _translate_text_uncached failed or didn't
    pass _structure_preserved.
    """
    if not text.strip():
        return text

    try:
        translated = GoogleTranslator(source='auto', target=lang_code).translate(text)
        if translated and not _looks_like_google_error_page(translated):
            return translated
        logger.warning("Google failed on one piece during the slow path - falling back to MyMemory for it")
    except Exception as e:
        logger.warning(f"Google raised on one piece during the slow path, falling back to MyMemory. Error: {e}")

    if len(text) <= MYMEMORY_CHUNK_LIMIT:
        return _translate_one_piece_mymemory(text, lang_code)
    pieces = _split_long_line(text, MYMEMORY_CHUNK_LIMIT)
    return " ".join(_translate_one_piece_mymemory(piece, lang_code) for piece in pieces)


def _translate_table_row(line: str, lang_code: str) -> str:
    """
    Translates a Markdown table row cell by cell and reassembles it with
    the exact same number of "|" characters in the exact same positions -
    see _translate_run for why cell-by-cell is what actually guarantees
    this, unlike sending the whole row as one sentence.
    """
    cells = line.split('|')
    translated_cells = []
    for cell in cells:
        content = cell.strip()
        if not content:
            translated_cells.append(cell)
            continue
        leading_ws = cell[:len(cell) - len(cell.lstrip())]
        trailing_ws = cell[len(cell.rstrip()):]
        translated_cells.append(leading_ws + _translate_run(content, lang_code) + trailing_ws)
    return '|'.join(translated_cells)


def _translate_line_preserving_structure(line: str, lang_code: str) -> str:
    """
    Routes one line through whichever handling keeps its Markdown
    structure intact: table rows cell-by-cell, headers with their "#"
    prefix split off and reattached, everything else as one plain run.
    """
    stripped_line = line.strip()

    if '|' in stripped_line and not _TABLE_SEPARATOR_RE.match(stripped_line):
        return _translate_table_row(line, lang_code)

    header_match = _HEADER_RE.match(stripped_line)
    if header_match:
        leading_ws = line[:len(line) - len(line.lstrip())]
        prefix, heading_text = header_match.groups()
        return leading_ws + prefix + _translate_run(heading_text, lang_code)

    return _translate_run(line, lang_code)


def translate_text(text: str, target_lang: str = "English") -> str:
    """
    Translates text into the requested language.

    English is returned completely untouched (bold and all). For every
    other language, see _translate_text_uncached for the fast-path-first,
    validated-fallback strategy.
    """
    lang_code = lang_map.get(target_lang, "en")

    if lang_code == "en":
        return text

    cache_key = _cache_key(text, target_lang)
    if cache_key in _translation_cache:
        return _translation_cache[cache_key]

    result = _translate_text_uncached(text, lang_code)
    _translation_cache[cache_key] = result
    return result


def _translate_text_uncached(text: str, lang_code: str) -> str:
    """
    The actual translation work, run only on a cache miss.

    STEP 1 - FAST PATH: one Google call on the whole answer, uncut. If it
    succeeds AND _structure_preserved says the Markdown skeleton survived
    intact, return it immediately - this is the common case and it's a
    single network round-trip.

    STEP 2 - SLOW BUT SAFE PATH: only reached if Google failed outright,
    returned empty/an error page, or its result didn't pass structural
    validation. Falls through to the line-by-line, structure-aware
    method - table rows cell-by-cell, headers with their "#" split off -
    where each individual piece tries Google first and MyMemory as its
    own fallback. If ANY piece fails on both engines, the whole answer
    reverts to English rather than shipping a mixed-language result.

    This means the person gets the fast single-call speed whenever
    Google happens to preserve the Markdown correctly, and only pays the
    slower per-piece cost on the specific answers where it doesn't -
    instead of either always risking a broken table (old fast-only
    version) or always paying the per-piece cost even when it wasn't
    needed (old slow-only version).
    """
    clean_text = _strip_bold_markers(text)
    clean_text = _normalize_typographic_chars(clean_text)

    # STEP 1: fast path.
    try:
        translated = GoogleTranslator(source='auto', target=lang_code).translate(clean_text)
        if (
            translated
            and not _looks_like_google_error_page(translated)
            and _structure_preserved(clean_text, translated)
        ):
            return translated
        logger.info(
            "Whole-answer Google translation failed, was empty, or didn't "
            "preserve Markdown structure - using the slower structure-"
            "preserving fallback for this answer."
        )
    except Exception as e:
        logger.warning(f"Whole-answer Google translation raised, falling back to the slower path. Error: {e}")

    # STEP 2: slow but safe path.
    lines = clean_text.split("\n")
    translated_lines = []

    try:
        for line in lines:
            stripped_line = line.strip()

            if not stripped_line:
                translated_lines.append(line)
                continue

            if _TABLE_SEPARATOR_RE.match(stripped_line):
                translated_lines.append(line)
                continue

            translated_lines.append(_translate_line_preserving_structure(line, lang_code))
    except _PieceTranslationFailed:
        logger.warning(
            "A piece could not be translated by either engine during the "
            "slow path - returning the answer in English rather than a "
            "mixed-language result."
        )
        return text

    return "\n".join(translated_lines)