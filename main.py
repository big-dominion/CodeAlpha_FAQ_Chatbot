"""
DOMINION LexOracle : CodeAlpha AI Edition
Main FastAPI Gateway and Orchestrator

Author: Samson Olawumi

WHY THIS FILE EXISTS:
This file is the front door of the app. It serves the webpage, receives
each chat message (typed or spoken), and calls out to the other files in
services/ to do the actual work: searching the statutes, asking the AI,
translating the answer, and turning it into speech. Keeping this file
focused only on "receive a request, call the right services, send back a
response" is why the AI logic itself lives in services/rag.py instead of
being written directly in here.

WHAT THIS FILE DOES (high-level summary, expanded):
This is the FastAPI application itself - the single process that the
webserver actually runs, and the only file in the project that exposes
HTTP endpoints. It ties together every other module in services/ into a
working request/response cycle, but deliberately contains none of their
internal logic itself:

- `services/rag.py` (query_lex_oracle) does the actual "answer the legal
  question" work: query expansion, vector retrieval, and LLM generation.
- `services/translator.py` (translate_text) turns an English answer into
  whichever language the user has selected.
- `services/audio_handler.py` (speech_to_text_from_bytes,
  text_to_speech_base64) converts between recorded voice audio and text
  in both directions.

Three HTTP endpoints are exposed, each rate-limited independently (see the
`@limiter.limit(...)` decorators) to keep the Groq/Pinecone usage this app
pays for bounded even under abusive or accidental repeated requests:

- `POST /api/v1/chat` - the main conversational endpoint. Accepts either
  typed text or a recorded audio clip, maintains a per-session
  conversation history in memory, calls `query_lex_oracle` to get a
  grounded English answer, translates it if needed, and returns both the
  translated and the original raw English answer (see the long comment
  above the endpoint's return statement for why raw_answer is sent back
  too).
- `POST /api/v1/tts` - generates speech audio (and updated display text)
  for a given piece of text in a given language, called on demand only
  when the user actually presses "Listen" on an answer, using whatever
  language is currently selected at that moment.
- `POST /api/v1/translate` - text-only re-translation, with no audio
  synthesis, used when the user changes the language dropdown after an
  answer is already visible, so every answer bubble on screen can be
  updated cheaply without paying for a TTS call each time.

Session state (`chat_sessions`) is a plain in-memory dict keyed by a
per-browser-session UUID, holding that session's conversation history.
It's cleaned up opportunistically (see `touch_session` below) rather than
on any dedicated schedule - this is fine for a single-process dev/small
deployment, but like the in-memory caches in translator.py and rag.py, it
would need to move to something shared like Redis if this app ever runs
as multiple worker processes or needs to survive restarts.

The module also fails fast at import time if required API keys
(`GROQ_API_KEY`, `PINECONE_API_KEY`) aren't set in the environment,
rather than starting up successfully and only failing later, confusingly,
on the first real request that needs them.

PROMPT-INJECTION / DRIFT DEFENSE (added):
This file now carries the second and third layers of a four-layer
defense against a user trying to get the model to produce code, SQL,
screenplay/roleplay content, or a leak of its own system prompt - the
first and second layers (the system prompt's ABSOLUTE BOUNDARIES block,
and the Groq `stop` sequences on the final answer call) live in
services/rag.py, since they need to sit next to the prompt and the Groq
call itself. This file adds:
  - `looks_suspicious` / `DRIFT_INTENT_SIGNALS`: a cheap, regex-only,
    logging-only triage check run on the INCOMING user message, before
    query_lex_oracle is even called. It never blocks anything - see its
    own comment for why a false positive here should cost nothing.
  - `contains_drift` / `DRIFT_PATTERNS`: a regex check run on the
    OUTGOING raw_answer, which is the backstop layer - it doesn't care
    how the model was tricked, only what actually came out, and it
    substitutes a fixed refusal sentence if the answer looks like code,
    SQL, a screenplay, or a prompt-leak attempt slipped through both of
    the earlier layers in rag.py.
"""

import os
import sys
import time
import uuid
import logging
import re

from dotenv import load_dotenv

# override=True means values in the local .env file take precedence over
# any same-named variable that might already exist in the actual process
# environment (e.g. left over from a parent shell) - this is what makes
# this the "first" load_dotenv() call other modules' own load_dotenv()
# calls (e.g. in services/vector_store.py, cloud_index.py) can safely
# treat as a no-op once this file has already run first as the app's
# entry point.
load_dotenv(override=True)

from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from services.audio_handler import text_to_speech_base64, speech_to_text_from_bytes
from services.translator import translate_text
from services.rag import query_lex_oracle

logger = logging.getLogger(__name__)
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))

# Fail fast at startup rather than at request time: if either of these
# keys is missing, every real request would eventually fail deep inside
# rag.py or vector_store.py anyway, but only after the app has already
# started and looked "up" to anyone checking on it. Checking here means
# a misconfigured deployment never even finishes starting, with a clear
# message naming exactly which variable(s) are missing.
REQUIRED_ENV = ["GROQ_API_KEY", "PINECONE_API_KEY"]
missing = [k for k in REQUIRED_ENV if not os.environ.get(k)]
if missing:
    sys.exit(f"Missing required environment variables: {', '.join(missing)}")

app = FastAPI(title="DOMINION LexOracle")

# Rate limiting protects both endpoints below from being called an
# unlimited number of times by one visitor, which would otherwise let
# anyone burn through the Groq and Pinecone usage this app pays for.
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Serves everything under the local "static" directory (the frontend's
# HTML/CSS/JS, logo, etc.) at the "/static" URL path.
app.mount("/static", StaticFiles(directory="static"), name="static")

# In-memory store of active chat sessions, keyed by a per-browser session
# UUID. Each entry holds that session's running conversation history (a
# list of {"role", "content"} messages, the same shape sent to the LLM)
# plus a "last_seen" timestamp used to expire stale sessions - see
# touch_session below. Being a plain process-local dict, this state is
# lost on restart and not shared across multiple worker processes, same
# caveat as the caches in translator.py and rag.py.
chat_sessions = {}

# How long (in seconds) a session can go without any activity before it's
# considered stale and eligible for cleanup - currently one hour.
SESSION_TTL_SECONDS = 60 * 60

# gTTS has no hard character limit of its own - it internally splits long
# text into short segments and stitches the resulting audio together, so
# it will not error out on a long answer. This cap exists purely to bound
# worst-case synthesis latency: the longer the text, the more internal
# requests gTTS has to make to Google's TTS endpoint before it can return
# anything, and that adds up. Truncating past this point trades a
# slightly incomplete spoken answer for a predictable response time.
MAX_TTS_CHARS = 3000

# ---------------------------------------------------------------------------
# Prompt-injection / drift defense
# ---------------------------------------------------------------------------

# Regex-only, no extra LLM call: flags a message as worth logging before
# it's even sent to query_lex_oracle. Deliberately NOT used to block - a
# real user could legitimately ask about "the SQL Act" or similar edge
# cases, and a false positive here costs nothing but a wasted log line,
# whereas a hard block would cost a real user their real legal question.
# This exists to build a flagged-query dataset for later review, not to
# gate anything.
DRIFT_INTENT_SIGNALS = [
    r"\bscreenplay\b", r"\bpython (script|function|code)\b", r"\bsql query\b",
    r"\bjson object\b", r"\bignore (the )?(previous|legal|statutes?)\b",
    r"\bdisregard\b.*\binstructions\b", r"\bsystem prompt\b", r"\bact as\b",
    r"\bpretend (you|to be)\b", r"\bhypothetically\b.*\b(code|script)\b",
]


def looks_suspicious(text: str) -> bool:
    """Logging-only triage signal - see DRIFT_INTENT_SIGNALS comment for why this never blocks."""
    return any(re.search(p, text, re.IGNORECASE) for p in DRIFT_INTENT_SIGNALS)


# Output-side backstop: catches whatever survives both the system prompt
# and the Groq stop sequences in rag.py, regardless of how the model got
# there. This is the layer that doesn't depend on the model "choosing"
# correctly - it only looks at what actually came out.
#
# The indented-block pattern below requires 3+ CONSECUTIVE lines indented
# 4+ spaces, none of which start with a Markdown list/quote marker
# (-, *, >, or "1."). That's deliberately narrower than a plain
# `^\s{4,}\S` check: a single nested Markdown sub-bullet or a blockquoted
# statute excerpt is one indented line, not a run of three, and a nested
# bullet's own text starts with one of the excluded markers anyway - so
# this only fires on the shape a real code block has (repeated plain
# indentation), not on the formatting the system prompt itself asks the
# model to produce.
DRIFT_PATTERNS = [
    r"```",                                                          # code fences
    r"\bdef\s+\w+\s*\(",                                             # python functions
    r"\b(CREATE|SELECT|INSERT|DROP)\s+(TABLE|\*|INTO)\b",            # SQL
    r"^\s*import\s+\w+",                                             # imports
    r"\bfunction\s*\(",                                              # JS
    r'"[a-z_]+"\s*:\s*"',                                            # raw JSON key:string pairs
    r"(?:^\s{4,}(?!-|\*|>|\d+\.|\([a-zA-Z]\))\S.*\n){3,}",           # 3+ consecutive plain-indented lines = code block, not a sub-bullet (also exempts lettered/numbered statute sub-clauses like (a), (b))
    r"\b(INT\.|EXT\.|FADE (IN|OUT)|PROSECUTOR|JUDGE)\b",              # screenplay markers
    r"\bmy (system prompt|instructions were|instructions are)\b",    # prompt leak attempt
]


def contains_drift(text: str) -> bool:
    return any(re.search(p, text, re.IGNORECASE | re.MULTILINE) for p in DRIFT_PATTERNS)


def touch_session(session_id: str):
    """
    Records activity on a session, opportunistically expires stale
    sessions, and returns the (possibly newly created) session's
    conversation history list.

    Detailed flow:
    - Compute the current time once.
    - Sweep ALL sessions currently in `chat_sessions` (not just the one
      being touched) and collect the IDs of any whose "last_seen" is
      older than SESSION_TTL_SECONDS ago, then delete every one of those
      - this is what actually reclaims memory from abandoned sessions,
      since nothing else in this file runs on a schedule to do it. It
      happens as a side effect of any session being touched, rather than
      via a separate background job/timer.
    - If `session_id` doesn't exist yet in `chat_sessions` (a brand new
      session), create it with an empty history list and the current
      timestamp.
    - Either way, update this session's "last_seen" to now, so it won't
      be swept up by the expiry check above until another full TTL
      period of inactivity passes.
    - Return this session's history list (a live reference, not a copy -
      callers like chat_endpoint append directly to it, which mutates
      the same list stored in chat_sessions).
    """
    now = time.time()
    expired = [sid for sid, s in chat_sessions.items() if now - s["last_seen"] > SESSION_TTL_SECONDS]
    for sid in expired:
        del chat_sessions[sid]

    if session_id not in chat_sessions:
        chat_sessions[session_id] = {"history": [], "last_seen": now}
    chat_sessions[session_id]["last_seen"] = now
    return chat_sessions[session_id]["history"]


@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    """
    Serves the main page fresh from disk on every request (already
    correct - this file is never stale on the SERVER side). What was
    missing was any Cache-Control header on the response: with none set,
    HTMLResponse sends nothing, and the browser falls back to its own
    heuristic caching for "/" - which is exactly what made a normal
    refresh keep showing an old version of the page until a hard refresh
    (which bypasses the browser cache entirely) forced a real re-fetch.
    Explicitly telling the browser not to cache this response means a
    plain refresh always re-fetches the current file, no hard refresh
    required. /static/* assets (logo, source-links.json) are untouched
    by this and keep their normal caching, since those change rarely and
    were never the source of the stale-page symptom.

    Detailed flow:
    - Open and read static/index.html fresh from disk on every single
      request (no in-memory caching of the file's content at all), so
      the server always serves whatever is currently on disk.
    - Return it as an HTMLResponse, explicitly attaching cache-prevention
      headers (Cache-Control, Pragma, Expires) so the BROWSER doesn't
      cache this particular response either - covering both the legacy
      Pragma header (for older caches) and the modern Cache-Control one.
    """
    with open("static/index.html", "r", encoding="utf-8") as f:
        html = f.read()
    return HTMLResponse(
        content=html,
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


# Any of these strings means the person wants their answer read in an
# English voice, just with a different accent. Checking with startswith
# instead of an exact match against the single word "English" is what
# actually matters here: once the language picker started offering
# "English (US)", "English (UK)", and so on instead of a single plain
# "English" entry, a plain equality check against "English" would have
# been false for every one of them, and would have sent perfectly good
# English text through the translator for no reason on every single
# request. The translator itself would have caught this and returned the
# text unchanged, so nothing actually broke, but this check is the
# correct, honest fix rather than relying on that safety net silently
# papering over a wrong condition.
def _is_english_variant(lang_label: str) -> bool:
    """
    Returns True if `lang_label` names any English accent/variant option
    the language picker offers (e.g. "English", "English (US)", "English
    (UK)", "English (Nigeria)"), by checking whether it simply starts
    with the word "English" - see the comment above this function for
    why a startswith check is used here instead of an exact-match
    comparison. Used throughout this file to decide whether translation
    needs to run at all before display/synthesis.
    """
    return lang_label.startswith("English")


@app.post("/api/v1/chat")
@limiter.limit("10/minute")
async def chat_endpoint(
    request: Request,
    session_id: str = Form(None),
    text_input: str = Form(None),
    audio_file: UploadFile = File(None),
    target_lang: str = Form("English (US)"),
    doc_filter: str = Form("all")
):
    """
    Main conversational endpoint - accepts either typed text or a
    recorded voice clip, gets a grounded answer from the RAG pipeline,
    translates it if needed, and returns the result along with the
    session's updated history.

    Detailed flow:
    - Resolve the session: if no `session_id` was sent (or it's the
      literal string "null", which can happen from a frontend that
      hasn't initialized one yet), generate a fresh UUID for this
      conversation.
    - Call `touch_session` to get this session's live history list
      (creating a new session entry if needed, and opportunistically
      expiring old ones as a side effect - see touch_session above).
    - Determine the actual user message:
        * Default to whatever was submitted as `text_input`.
        * If an `audio_file` was ALSO provided, read its raw bytes and
          run them through `speech_to_text_from_bytes` (using the
          currently selected `target_lang` so recognition uses the right
          locale), OVERWRITING `user_message` with the transcript - so
          audio takes priority over any accompanying text_input if both
          happen to be present.
    - If there's still no usable message at all (empty text, and either
      no audio was sent or the transcription came back empty), return a
      400 error immediately rather than proceeding with nothing to answer.
    - Log (never block on) a suspicious-intent signal against the
      resolved message via `looks_suspicious` - see that function's
      comment for why this is logging-only.
    - Append the user's message to this session's history as a
      {"role": "user", ...} entry - this mutates the same list object
      touch_session returned, which lives inside chat_sessions.
    - Call `query_lex_oracle(user_message, doc_filter, history)` to run
      the full RAG pipeline and get back the raw English answer plus its
      citations.
    - If the raw answer came back empty/blank (can genuinely happen after
      a very short or unclear voice recording, or an LLM hiccup),
      substitute a friendly fallback message asking the user to rephrase,
      rather than sending an empty bubble to the frontend. Otherwise, if
      the raw answer matches `contains_drift` (a code/SQL/screenplay/
      prompt-leak shape slipped past both defense layers in rag.py),
      substitute a fixed refusal sentence instead - this is the output-
      side backstop described in the module docstring.
    - Translate the answer for display, only if `target_lang` is NOT an
      English variant (per `_is_english_variant`) - otherwise
      `final_answer` stays exactly equal to the raw English answer, with
      no translation call made at all.
    - Append the assistant's FINAL (possibly translated) answer to the
      session history, so future turns in this conversation see what the
      user actually saw, in whatever language it was shown in.
    - Return a JSON payload with: the session ID (so the frontend can
      keep reusing it on subsequent requests), the resolved user message
      (useful when it came from audio transcription, so the frontend can
      display what was actually understood), the final display answer,
      the ORIGINAL raw (English, untranslated) answer, the citations
      list, and the full updated history.
    - See the long comment further below (right above the return
      statement in the original code) for exactly why raw_answer is sent
      back alongside the translated final_answer: it's what lets the
      later /api/v1/tts and /api/v1/translate calls always start
      translating fresh from a clean English source, no matter what
      language is picked after the fact, instead of translating an
      already-translated string.
    """
    if not session_id or session_id == "null":
        session_id = str(uuid.uuid4())

    history = touch_session(session_id)
    user_message = text_input

    if audio_file:
        audio_bytes = await audio_file.read()
        user_message = speech_to_text_from_bytes(audio_bytes, lang=target_lang)

    if not user_message:
        return JSONResponse({"error": "No input detected"}, status_code=400)

    if looks_suspicious(user_message):
        logger.warning(f"Suspicious query flagged: {user_message!r}")

    history.append({"role": "user", "content": user_message})

    response_data = query_lex_oracle(user_message, doc_filter, history)
    raw_answer = response_data.get("raw_answer", "Error generating response.")

    # Guards against a blank AI answer (this can happen after a very
    # short or unclear voice recording) reaching the person as an empty
    # message instead of a real sentence.
    if not raw_answer or not raw_answer.strip():
        raw_answer = "I wasn't able to generate a response to that. Could you rephrase your question?"
    elif contains_drift(raw_answer):
        raw_answer = "I can only provide statutory legal information from the source documents, and can't generate code, scripts, structured data formats, or fictional/roleplay content."

    citations = response_data.get("citations", [])

    final_answer = raw_answer
    if not _is_english_variant(target_lang):
        final_answer = translate_text(raw_answer, target_lang=target_lang)

    history.append({"role": "assistant", "content": final_answer})

    # No audio is generated here. The Listen button on the frontend calls
    # /api/v1/tts on demand instead, using whatever language is selected
    # at the moment it is clicked, not the language that was active when
    # this reply was first generated. This matters because a person can
    # change the language dropdown after an answer has already appeared,
    # and the audio should follow that later choice. raw_answer (the
    # original, untranslated English text) is sent back so that endpoint
    # can translate fresh from a clean English source no matter which
    # language is picked afterward, instead of translating a translation.
    return {
        "session_id": session_id,
        "user_message": user_message,
        "response_text": final_answer,
        "raw_answer": raw_answer,
        "citations": citations,
        "history": history
    }


# Request body shape for POST /api/v1/tts - a plain FastAPI/pydantic model
# so the framework validates and parses the incoming JSON automatically.
class TTSRequest(BaseModel):
    text: str
    lang: str = "English (US)"


@app.post("/api/v1/tts")
@limiter.limit("20/minute")
async def tts_endpoint(request: Request, body: TTSRequest):
    """
    Generates speech, and the matching translated text, for whichever
    language is requested at the moment this is called.

    Why this returns display_text as well as audio: if the person changes
    the language dropdown after an answer is already on screen and then
    presses Listen, the visible text on screen should update to match the
    new language too, not just the audio. Sending both back together
    means the frontend can update what is shown and what is heard at the
    same time, instead of the two getting out of sync.

    Why text is capped at MAX_TTS_CHARS before translation and synthesis:
    an unusually long answer would still work fine, gTTS just takes
    longer the more text it has to process internally. Truncating here,
    before either translate_text or text_to_speech_base64 is called,
    keeps both steps bounded instead of only capping one of the two.

    Detailed flow:
    - Strip the incoming text; if nothing's left, return a 400 error.
    - If the (stripped) text is longer than MAX_TTS_CHARS, truncate it
      down to that length BEFORE doing anything else with it - this
      bounds both the translation step and the TTS step that follow,
      rather than only capping one of them.
    - If the requested language is not an English variant (per
      `_is_english_variant`), translate the (possibly truncated) text
      into it via `translate_text`, producing `display_text`. Otherwise,
      `display_text` is simply the truncated English text unchanged.
    - Pass `display_text` (i.e. whatever language it actually ends up
      in) to `text_to_speech_base64` to synthesize the audio, receiving
      back both the base64-encoded audio string and an `available` flag
      indicating whether synthesis actually succeeded for this language.
    - Return `display_text`, `audio_base64`, and `available` together, so
      the frontend can update the visible text and play the audio (or
      show an honest "no audio available" state) in one coordinated
      update.
    """
    text = body.text.strip()
    if not text:
        return JSONResponse({"error": "No text provided"}, status_code=400)

    if len(text) > MAX_TTS_CHARS:
        text = text[:MAX_TTS_CHARS]

    display_text = text
    if not _is_english_variant(body.lang):
        display_text = translate_text(text, target_lang=body.lang)

    audio_base64, available = text_to_speech_base64(display_text, lang=body.lang)

    return {
        "display_text": display_text,
        "audio_base64": audio_base64,
        "available": available
    }


# Request body shape for POST /api/v1/translate.
class TranslateRequest(BaseModel):
    text: str
    lang: str = "English (US)"


@app.post("/api/v1/translate")
@limiter.limit("30/minute")
async def translate_endpoint(request: Request, body: TranslateRequest):
    """
    Translates text only, with no audio synthesis. Used when the language
    dropdown changes after an answer is already on screen, so every
    visible answer bubble can be retranslated immediately without paying
    the extra latency of a gTTS call for each one - audio is only
    generated later, on demand, when Listen is actually pressed.

    Detailed flow:
    - Strip the incoming text; if nothing's left, return a 400 error.
    - If the requested language is not an English variant, translate the
      text via `translate_text` into `display_text`; otherwise
      `display_text` is just the stripped text unchanged.
    - Return only `display_text` - no audio fields at all, unlike
      /api/v1/tts, since this endpoint's entire purpose is to be the
      cheaper, audio-free alternative for a language switch that hasn't
      (yet) been followed by a Listen press.
    """
    text = body.text.strip()
    if not text:
        return JSONResponse({"error": "No text provided"}, status_code=400)

    display_text = text
    if not _is_english_variant(body.lang):
        display_text = translate_text(text, target_lang=body.lang)

    return {"display_text": display_text}