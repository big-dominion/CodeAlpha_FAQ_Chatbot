"""
DOMINION LexOracle: Audio Handler
---------------------------------

WHAT THIS FILE DOES (high-level summary):
This module is the bridge between "raw audio/text" and "text/audio" for the
LexOracle app's voice features. It has two jobs, running in opposite
directions:

1. TEXT -> SPEECH (TTS): Takes a string of text (e.g. the AI's reply) and a
   requested language/accent, and turns it into playable audio using
   Google's gTTS service. The result is returned as a base64-encoded string
   so it can be sent straight to a web frontend and played in the browser
   with no temporary files written to disk.

2. SPEECH -> TEXT (STT): Takes raw audio bytes recorded by a user's browser
   microphone (WebM/Opus, from the browser's MediaRecorder API), converts
   that audio into a WAV file in memory, and sends it to Google's speech
   recognition service to get back a plain text transcript.

Everything in this file operates on in-memory byte buffers (BytesIO) rather
than files on disk. This keeps the app stateless and avoids leaving audio
artifacts behind on the server between requests.

Key supporting pieces:
- `imageio_ffmpeg` supplies an actual ffmpeg binary bundled inside a Python
  package, so `pydub` (used for audio format conversion) has something to
  call without requiring the host machine to have ffmpeg installed
  separately.
- `_get_supported_langs()` caches a live lookup of which languages gTTS
  actually supports, so we don't call the network on every single TTS
  request, but also don't get permanently stuck if that one lookup fails.
- `LANG_CODE_MAP` / `ENGLISH_TLDS` / the STT `lang_map` all translate the
  app's human-friendly language names (e.g. "Yoruba", "English (Nigeria)")
  into the specific codes that gTTS and Google's speech recognizer expect.
- Both `text_to_speech_base64()` and `speech_to_text_from_bytes()` are
  written to fail *quietly* (returning an empty/blank result) rather than
  raising exceptions, so that a bad recording or an unsupported language
  degrades gracefully instead of crashing the whole chat request.

If you're extending this file: the two public entry points other modules
should call are `text_to_speech_base64(text, lang)` and
`speech_to_text_from_bytes(audio_bytes, lang)`. `convert_to_wav()` and
`_get_supported_langs()` are internal helpers used by those two.
"""
import base64
from io import BytesIO
from gtts import gTTS
from gtts.lang import tts_langs
import speech_recognition as sr
from pydub import AudioSegment
import imageio_ffmpeg

# pydub does not decode audio itself. It hands the job to ffmpeg and reads
# the result back. imageio_ffmpeg bundles a real ffmpeg binary inside a
# normal Python package, so we point pydub at that instead of requiring
# ffmpeg to be installed separately on whatever machine runs this app.
AudioSegment.converter = imageio_ffmpeg.get_ffmpeg_exe()

# Module-level cache for gTTS's supported-languages lookup. Starts as None
# (meaning "never fetched yet"). Once a successful fetch happens, this holds
# the real dict for the rest of the process's lifetime. It is intentionally
# NOT set to an empty dict on failure — see _get_supported_langs() below for
# why that distinction matters.
_SUPPORTED_LANGS_CACHE = None


def _get_supported_langs() -> dict:
    """
    Fetches gTTS's real, current list of supported languages once, then
    reuses that same list for every future call instead of asking again.

    Why a failure is not cached: this list comes from a live network
    call. If that call fails once, perhaps from a brief network hiccup,
    we do not want the app to be stuck believing no language is
    supported for the rest of its uptime. Returning an empty dict on
    failure, without saving it into the cache variable, means the very
    next call will simply try fetching the real list again.

    Detailed flow:
    - If the cache is still None (nothing fetched yet, or the last attempt
      failed and left it untouched), try calling tts_langs(), which hits
      Google's servers for the current list.
    - On success, store that dict in the module-level cache so subsequent
      calls skip the network entirely.
    - On any exception, return an empty dict for THIS call only, leaving
      the cache as None so the next call retries the fetch from scratch.
    - If the cache is already populated, just hand back the cached dict
      immediately with no network call at all.
    """
    global _SUPPORTED_LANGS_CACHE
    if _SUPPORTED_LANGS_CACHE is None:
        try:
            _SUPPORTED_LANGS_CACHE = tts_langs()
        except Exception:
            return {}
    return _SUPPORTED_LANGS_CACHE


def convert_to_wav(audio_bytes: bytes) -> BytesIO:
    """
    Turns browser recorded audio into a WAV file, kept in memory.

    Why this exists: the mic button in index.html uses the browser's
    MediaRecorder, which produces WebM audio using the Opus codec by
    default. The speech_recognition library can only read WAV, AIFF, or
    FLAC files, so every recording has to be converted before it can be
    transcribed. This happens entirely in memory, so no audio file is
    ever written to disk.

    Detailed flow:
    - Wrap the raw incoming bytes in a BytesIO so pydub can treat them
      like a file-like object without touching disk.
    - Tell pydub to decode it explicitly as "webm" (rather than letting it
      guess), since that's the format the browser's MediaRecorder emits.
    - Export the decoded audio back out as WAV into a fresh in-memory
      buffer.
    - Rewind that buffer's read position to the start (seek(0)) so
      whatever consumes it next (speech_recognition's AudioFile) reads
      from the beginning rather than the end.
    """
    audio = AudioSegment.from_file(BytesIO(audio_bytes), format="webm")
    wav_io = BytesIO()
    audio.export(wav_io, format="wav")
    wav_io.seek(0)
    return wav_io


# Maps the app's human-readable language names to the two-letter language
# codes gTTS expects for its `lang` parameter. English is deliberately left
# out here — it's handled separately below via ENGLISH_TLDS, because English
# additionally needs an accent/TLD choice, not just a language code.
LANG_CODE_MAP = {
    "Yoruba": "yo",
    "Hausa": "ha",
    "Igbo": "ig",
    "French": "fr"
}

# gTTS does not check tld against any fixed, official list. It is simply
# the Google Translate web address the request is sent to
# (translate.google.<tld>), so any value here is safe to try. Whether it
# actually produces an audibly different accent depends entirely on what
# Google's servers do with that address, not on anything in this code.
#
# "com.ng" for Nigerian English is not on gTTS's own documented list of
# accent TLDs. It is included here as a genuine experiment rather than a
# guarantee: if Google happens to serve a distinct Nigerian sounding
# voice from that address, this will pick it up automatically. If it
# does not, the broad except in text_to_speech_base64 below catches the
# failure the same safe way it catches any other TTS problem, so trying
# it cannot break anything, it can only work or quietly fall back to no
# audio for that one message.
#
# South Africa ("co.za") was tried and removed. In practice it produced
# a voice that sounded the same as the UK option, so keeping both only
# added a confusing extra choice without a real difference to offer.
#
# Maps each English "variant" option shown to the user to the Google
# Translate country-domain (tld) that gTTS should send the request to.
ENGLISH_TLDS = {
    "English": "us",
    "English (US)": "us",
    "English (UK)": "co.uk",
    "English (Nigeria)": "com.ng"
}


def text_to_speech_base64(text: str, lang: str = 'English') -> tuple:
    """
    Converts text into speech and returns it as a base64 string the
    frontend can play directly, with no audio file saved anywhere.

    Why this returns a tuple of (audio, available) instead of just the
    audio string: the frontend needs to be able to tell apart two very
    different situations that used to look identical, "there is no audio
    because there was nothing to say" and "there is no audio because this
    language genuinely has no voice at all in gTTS". The available flag
    makes that difference explicit, so the frontend can show an honest
    message instead of a Listen button that quietly does nothing.

    Why the empty text check exists: if the AI ever returns an empty or
    blank answer, gTTS refuses to generate audio for nothing and raises
    an error. Checking for empty text up front avoids ever calling gTTS
    with nothing to say, so that crash cannot happen.

    Why the except catches "Exception" broadly: gTTS raises different
    error types for different failures (an AssertionError for empty text,
    other errors for network or language problems). Catching the general
    Exception class means any of these degrade to "no audio" instead of
    crashing the whole response, whatever type of error it happens to be.

    Detailed flow:
    - Bail out immediately with ("", False) if there's no real text to
      speak (covers None, empty string, and whitespace-only strings).
    - Fetch the (possibly cached) list of gTTS-supported languages.
    - Work out which gTTS language code and tld to use:
        * If `lang` is one of the recognized English variants, force the
          gTTS language code to "en" and pick the matching accent tld
          from ENGLISH_TLDS.
        * Otherwise, look up the code in LANG_CODE_MAP (defaulting to
          "en" if it's an unrecognized name) and use the generic "com"
          tld, since non-English languages don't have the same
          accent-via-tld trick applied here.
    - If we successfully loaded a real supported-languages list AND this
      language code isn't in it, short-circuit and return ("", False)
      rather than making a doomed request.
    - Otherwise, attempt the actual TTS request: build the gTTS object,
      write the resulting MP3 audio into an in-memory buffer, rewind it,
      and return it base64-encoded along with True.
    - If anything goes wrong during that attempt, log it and return
      ("", False) instead of letting the exception propagate.
    """
    if not text or not text.strip():
        return "", False

    supported = _get_supported_langs()

    if lang in ENGLISH_TLDS:
        gtts_lang = "en"
        tld = ENGLISH_TLDS[lang]
    else:
        gtts_lang = LANG_CODE_MAP.get(lang, "en")
        tld = "com"

    # If we do have a real supported-languages list from Google, and this
    # language genuinely is not on it, there is no point even trying,
    # since gTTS is certain to fail. If the list itself failed to load
    # (supported is empty), we do not block on that, we just attempt the
    # request anyway and let the try block below catch any real failure.
    if supported and gtts_lang not in supported:
        return "", False

    try:
        tts_fp = BytesIO()
        tts = gTTS(text=text, lang=gtts_lang, tld=tld, slow=False)
        tts.write_to_fp(tts_fp)
        tts_fp.seek(0)
        return base64.b64encode(tts_fp.read()).decode("utf-8"), True
    except Exception as e:
        print(f"TTS Error: {e}")
        return "", False


def speech_to_text_from_bytes(audio_bytes: bytes, lang: str = 'English') -> str:
    """
    Turns a recorded voice clip into text.

    Why the except is broad and just returns an empty string: Google's
    speech recognition service does not support every language equally
    well, and a bad or silent recording can also fail to transcribe. If
    this raised an error instead of failing quietly, one bad recording
    would crash the entire chat request. Returning an empty string lets
    the calling code treat it the same as "no input detected" and
    respond to the person normally instead of showing a server error.

    Detailed flow:
    - Build a local mapping from the app's human-readable language names
      to the locale codes Google's speech recognition API expects (e.g.
      "English (Nigeria)" -> "en-NG"). This is a separate map from the
      TTS ones above because STT locale codes and TTS language/tld
      combinations don't line up one-to-one.
    - Look up the requested `lang`, defaulting to "en-US" for anything
      unrecognized.
    - Create a fresh SpeechRecognition Recognizer instance for this call.
    - Convert the incoming raw bytes to an in-memory WAV file via
      convert_to_wav(), since speech_recognition can't read WebM/Opus
      directly.
    - Load that WAV data as an AudioFile source, record the entire clip
      into an AudioData object, and send it to Google's hosted
      recognizer (recognize_google) along with the chosen locale code.
    - Return whatever transcript text comes back.
    - If any step fails (bad audio, network issue, unsupported language,
      silence with nothing recognizable, etc.), log the error and return
      an empty string rather than raising.
    """
    lang_map = {
        "English": "en-US", "English (US)": "en-US", "English (UK)": "en-GB",
        "English (Nigeria)": "en-NG",
        "Yoruba": "yo", "Hausa": "ha", "Igbo": "ig", "French": "fr-FR"
    }
    google_lang = lang_map.get(lang, "en-US")

    recognizer = sr.Recognizer()
    try:
        wav_audio = convert_to_wav(audio_bytes)
        with sr.AudioFile(wav_audio) as source:
            audio_data = recognizer.record(source)
            return recognizer.recognize_google(audio_data, language=google_lang)
    except Exception as e:
        print(f"STT Error: {e}")
        return ""