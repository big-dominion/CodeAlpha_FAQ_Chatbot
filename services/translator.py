"""
DOMINION LexOracle: Translation Service
---------------------------------------

WHAT THIS FILE DOES:
Translates the AI's English answer into whichever language the user picked
(Yoruba, Hausa, Igbo, French, etc.). Public entry point: translate_text().
For translating several independent texts as one coordinated batch (e.g.
every bubble that needs updating on a single language-switch event), see
translate_many() at the bottom of this file.

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

CONCURRENCY (added): the slow path used to translate every line one at a
time, in a plain `for` loop - a legal answer with several headers and a
table regularly has 15-30+ lines/cells needing their own network round
trip, and translating them sequentially meant paying the full latency of
every single one, back to back, before the answer was ready (observed in
practice taking many seconds for a single answer). Since each of these is
a blocking network call and not CPU work, they don't need to happen one
after another - `_translate_text_uncached`'s slow path now fires all of a
line's translations at once across a small thread pool
(`_SLOW_PATH_MAX_WORKERS`) and waits for them together, so the wall-clock
cost is roughly the slowest single call rather than the sum of all of
them. See the comment above `_mymemory_lock` for how the MyMemory
per-second rate limit is still respected under this concurrency.

BATCHING ACROSS BUBBLES (added): even with the per-answer concurrency
above, a language switch on a multi-bubble conversation used to call
translate_text() once per bubble, back to back - each call potentially
spinning up its OWN thread pool and firing its OWN burst of requests on
top of whatever the previous bubble's call was still doing. MyMemory's
5-requests/second cap is a flat per-second ceiling that registering
MYMEMORY_EMAIL does NOT raise (email only raises the daily word quota),
so several bubbles' worth of concurrent slow-path bursts landing in the
same second is what was actually triggering "too many requests" errors -
not a lack of email registration. translate_many() is the fix: it takes
a LIST of texts for one language-switch event, checks the cache for all
of them up front, and dispatches only the uncached ones across ONE shared
thread pool - so N bubbles never multiply into N independent bursts, they
share the one real budget the throttle enforces.

Everything else - bold-stripping, typographic normalization, MyMemory's
garbage-result detection, the whole-answer-reverts-to-English-on-any-piece-
failure guarantee, and the translation cache - is unchanged from before.
"""
import logging
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
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
#
# IMPORTANT: this raises the DAILY word quota only. It does NOT raise
# MyMemory's separate 5-requests/second cap (see _throttle_mymemory
# below) - that per-second limit applies identically whether or not an
# email is registered, which is why "too many requests" errors can still
# happen even with MYMEMORY_EMAIL set. The real fix for THAT is reducing
# how many concurrent requests get fired in the same second - see
# translate_many() at the bottom of this file.
MYMEMORY_EMAIL = os.environ.get("MYMEMORY_EMAIL")

# MyMemory's free tier rejects requests longer than roughly 500 characters.
# 450 leaves some room below that hard limit, so a line that is close to
# the edge does not get rejected by a small miscount.
MYMEMORY_CHUNK_LIMIT = 450

# How many lines/cells the slow path translates at once. Each unit is a
# blocking network call, not CPU work, so this is deliberately modest
# rather than "as many as there are lines" - Google's scrape endpoint is
# unofficial and has no documented concurrency allowance, so a small pool
# (a handful of simultaneous requests) gets most of the latency win
# without looking like a burst that risks a temporary block. MyMemory
# calls made from inside this pool are still paced to its real per-second
# cap by `_mymemory_lock` below, independent of this worker count. Also
# reused by translate_many() below as the shared pool size across an
# entire batch of bubbles, rather than per-bubble.
_SLOW_PATH_MAX_WORKERS = 6

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
# (distinct from its much larger daily word quota, and NOT raised by
# registering MYMEMORY_EMAIL - see the comment above that variable).
# Translating a structured answer header-by-header and table-cell-by-cell,
# when the slower fallback path is used, can easily fire off many
# MyMemory requests for a single answer in quick succession - comfortably
# over that per-second cap if nothing paces them. _throttle_mymemory is
# called immediately before every MyMemory request and sleeps just long
# enough to keep consecutive calls at least this far apart.
#
# _mymemory_lock makes this safe now that multiple slow-path lines - and,
# via translate_many() below, multiple whole BUBBLES - can reach
# _translate_one_piece_mymemory from different threads at once (see
# _SLOW_PATH_MAX_WORKERS above): without the lock, two threads could both
# read the same stale _last_mymemory_call_at, both compute a "wait" based
# on it, and both fire immediately after their own sleep - letting
# concurrency quietly defeat the pacing this function exists to enforce.
# The lock only serializes the MyMemory dispatch moment itself (read
# last-call-time, sleep if needed, update last-call-time); it does not
# serialize Google calls, which make up the large majority of slow-path
# traffic and are unaffected by this.
_MYMEMORY_MIN_INTERVAL_SECONDS = 0.22
_last_mymemory_call_at = 0.0
_mymemory_lock = threading.Lock()


def _throttle_mymemory():
    global _last_mymemory_call_at
    with _mymemory_lock:
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

    Called once per non-trivial line from the slow path in
    _translate_text_uncached, potentially from several worker threads at
    once - see _SLOW_PATH_MAX_WORKERS. This function and everything it
    calls (_translate_run, _translate_table_row, _translate_one_piece_mymemory)
    touch no shared mutable state directly, so no locking is needed here;
    the only shared state on this path is the MyMemory throttle, which
    guards itself via _mymemory_lock.
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
    Translates a SINGLE piece of text into the requested language. For
    translating several bubbles at once as part of one language-switch
    event, use translate_many() instead - see its docstring at the bottom
    of this file for why that matters for MyMemory's rate limit.

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

    Every non-trivial line's translation is dispatched to a small thread
    pool up front (see _SLOW_PATH_MAX_WORKERS and the module docstring's
    "CONCURRENCY" section) instead of one line at a time, so the total
    wait is close to the slowest single line rather than the sum of every
    line. Blank lines and table separator rows are never submitted to the
    pool at all - there's nothing in them to translate, so they're just
    copied straight into the result in their original position.

    This means the person gets the fast single-call speed whenever
    Google happens to preserve the Markdown correctly, and only pays the
    slower per-piece cost - now parallelized - on the specific answers
    where it doesn't, instead of either always risking a broken table
    (old fast-only version) or always paying the full sequential
    per-piece cost even when it wasn't needed (old slow-only version).
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

    # STEP 2: slow but safe path - now parallelized across lines/cells.
    lines = clean_text.split("\n")
    translated_lines = [None] * len(lines)
    work_items = []  # (index, line) pairs that actually need translation

    for i, line in enumerate(lines):
        stripped_line = line.strip()

        if not stripped_line or _TABLE_SEPARATOR_RE.match(stripped_line):
            # Nothing to translate - passed through untouched, and never
            # submitted to the thread pool below.
            translated_lines[i] = line
        else:
            work_items.append((i, line))

    if not work_items:
        return "\n".join(translated_lines)

    executor = ThreadPoolExecutor(max_workers=min(_SLOW_PATH_MAX_WORKERS, len(work_items)))
    try:
        futures = {
            executor.submit(_translate_line_preserving_structure, line, lang_code): i
            for i, line in work_items
        }
        for future in futures:
            i = futures[future]
            # .result() re-raises _PieceTranslationFailed here if that
            # particular line's translation failed on both engines - the
            # except block below catches it and reverts the whole answer
            # to English, same guarantee as the old sequential loop had.
            translated_lines[i] = future.result()
    except _PieceTranslationFailed:
        logger.warning(
            "A piece could not be translated by either engine during the "
            "slow path - returning the answer in English rather than a "
            "mixed-language result."
        )
        # Any other lines still running in the pool are no longer needed
        # once we've decided to revert the whole answer to English -
        # cancel_futures skips them instead of waiting for them to finish
        # before this function can return.
        executor.shutdown(wait=False, cancel_futures=True)
        return text
    else:
        executor.shutdown(wait=True)

    return "\n".join(translated_lines)


def translate_many(texts: list, target_lang: str = "English") -> list:
    """
    Translates multiple independent texts (e.g. every bubble that needs
    translating for one language-switch event) as ONE coordinated batch,
    sharing a single cache pass and a single thread pool across all of
    them - instead of each text calling translate_text() separately and
    each one spinning up its own _SLOW_PATH_MAX_WORKERS-sized pool.

    Why this exists: MyMemory's 5-requests/second cap is a flat per-second
    ceiling that MYMEMORY_EMAIL does NOT raise (email only raises the
    daily word quota - see the comment above that variable). The actual
    fix for "too many requests" errors is never sending more concurrent
    pieces than the shared throttle can absorb in the first place, not
    registering an email. This function is the coordination point: every
    text's slow-path work (if any) goes through the SAME pool and the
    SAME module-level _mymemory_lock/_throttle_mymemory, so the real cap
    is respected across the whole batch, not just within one answer.

    Flow:
    - English requested -> return texts unchanged, no work at all.
    - For each text, check the cache first; only texts that miss go on
      to real translation work.
    - If every text was cached, return immediately - zero network calls.
    - Otherwise, dispatch translation for every uncached text across one
      shared ThreadPoolExecutor sized to _SLOW_PATH_MAX_WORKERS (not
      per-text), so N texts don't multiply into N separate pools each
      trying to claim that many workers simultaneously.
    - Each completed translation is written into both the cache and the
      results list before returning, in the same order as the input
      `texts` list (not completion order), so callers can zip results
      back onto whatever they used to build the request.
    """
    lang_code = lang_map.get(target_lang, "en")

    if lang_code == "en":
        return list(texts)

    results = [None] * len(texts)
    uncached_indices = []

    for i, text in enumerate(texts):
        cache_key = _cache_key(text, target_lang)
        if cache_key in _translation_cache:
            results[i] = _translation_cache[cache_key]
        else:
            uncached_indices.append(i)

    if not uncached_indices:
        return results

    def _do_one(i):
        text = texts[i]
        translated = _translate_text_uncached(text, lang_code)
        _translation_cache[_cache_key(text, target_lang)] = translated
        return i, translated

    worker_count = min(_SLOW_PATH_MAX_WORKERS, len(uncached_indices))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(_do_one, i) for i in uncached_indices]
        for future in futures:
            i, translated = future.result()
            results[i] = translated

    return results