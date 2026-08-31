# DOMINION LexOracle

**CodeAlpha AI Internship Project**

DOMINION LexOracle is a legal question answering tool for Nigerian statutory law. You ask it a question in plain English (or Yoruba, Hausa, Igbo, or French), and it searches through six real Nigerian legal documents, finds the sections that actually answer your question, and gives you an answer that quotes those exact sections. It never makes up an answer from general knowledge. If the statutes do not cover something, it says so directly.

**Live demo:** https://dominion-lexoracle.onrender.com/ 

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Statutes Covered](#statutes-covered)
- [Architecture Notes](#architecture-notes)
- [Tech Stack](#tech-stack)
- [Full Project Structure](#full-project-structure)
- [Prerequisites](#prerequisites)
- [Installation and Setup](#installation-and-setup)
- [Environment Variables](#environment-variables)
- [Running the Application](#running-the-application)
- [Populating the Vector Database](#populating-the-vector-database)
- [API Reference](#api-reference)
- [Known Limitations](#known-limitations)
- [Deployment](#deployment)
- [Acknowledgements](#acknowledgements)

## Overview

The project has two halves that work together:

1. A frontend (a single HTML page) where you type or speak your question, pick a language, and see the answer with clickable citations.
2. A backend (built with FastAPI) that takes your question, searches a database of legal text for the most relevant sections, sends those sections to an AI model along with your question, and sends the AI's grounded answer back to you.

The AI is never allowed to answer from what it already knows about Nigerian law. It is only allowed to answer using the exact statute text it was handed for that specific question. This is what makes the answers trustworthy: every claim can be traced back to a real section of a real law.

## Key Features

- Statutory search grounded in real law: every answer is backed by actual retrieved text from the Constitution, CAMA, the Evidence Act, ACJA, the Criminal Code, or the Penal Code, not the AI's memory.
- Clickable citations: every answer shows which sections it used, with a match score and the ability to open the exact statute text in a popup, or the full original document.
- Multilingual answers: ask or read answers in English, Yoruba, Hausa, Igbo, or French. Google Translate is used first, with an automatic backup translator if Google is unavailable.
- Consistent translations: if part of a translation fails, the whole answer falls back to English instead of showing a confusing mix of two languages in one answer.
- Voice input and output: you can speak your question instead of typing it, and you can have any answer read aloud, with a live transcript shown back to you so a bad recording can be caught before it is answered.
- Document filtering: you can limit a search to just one law (for example, only CAMA) or search across all six at once.
- Consistent answers: asking the exact same question twice gives you the same retrieved sections and the same answer, instead of a random result each time.
- Session memory: the app remembers your recent questions in the same conversation, so follow-up questions make sense.
- Locked language switching: while an answer is being translated into a new language, the language picker is disabled so a second selection cannot interrupt or race the first one.
- Zero-build frontend: the entire user interface lives in one static index.html file. There is no npm install and no build pipeline required.

## Statutes Covered

- Constitution of the Federal Republic of Nigeria, 1999
- Companies and Allied Matters Act (CAMA), 2020
- Evidence Act 2011 (Consolidated with 2023 Amendments)
- Administration of Criminal Justice Act (ACJA), 2015
- Criminal Code
- Penal Code

## Architecture Notes

A few design choices in this project were made on purpose, and it helps to understand why before reading the code.

**The AI is only allowed to answer from retrieved text, never from memory.** The system prompt sent to the AI model on every request explicitly forbids it from using outside knowledge. It is told to answer only from the statute sections it was handed, to say plainly what is missing if those sections only partly cover the question, and to say "I cannot find the answer in the provided statutes" if they do not cover it at all. This is what makes the tool trustworthy for a legal use case: a wrong or missing citation is much worse than an honest "I do not know."

**The user's question is searched in two forms, not one.** Before searching the database, the raw question is expanded by a quick AI call into better legal search terms. Both the original question and the expanded version are searched, and the results are merged. This exists because AI generated text is not perfectly identical between two otherwise identical requests, even with the model's temperature set to zero. A slightly different expansion could occasionally miss a critical section. Searching both versions and combining the results means a section only has to be found by one of the two phrasings, not both.

**The same question is cached so it always returns the same answer.** The very first version of this app occasionally gave two different answers, or two different sets of citations, to the exact same question asked twice. This traced back to the AI's own request expansion step producing a slightly different search phrase each time, which then pulled a slightly different set of statute sections from the database. Caching the expansion by the exact question text fixes this: the same question always expands the same way, always searches the same way, and always answers the same way.

**Repeated opening questions are served from a small cache, without waiting on the AI or the database.** Separately from the search-expansion cache described above, the very first question of a new conversation is checked against a bounded, 500-entry cache before anything else runs. During a burst of traffic, many different people tend to ask the same handful of common questions when they first land on the app - caching those means every repeat after the first one costs nothing: no AI call, no database search, no wait. This only applies to a conversation's first message, and only when searching all six laws at once, since a later message depends on that specific conversation's history and can't safely be shared across different users.

**A single, short retry protects against a momentary AI service hiccup.** The final call to the AI model is retried once, after a brief pause, if it fails outright. This is capped at exactly one retry with a short fixed wait, rather than several attempts with a growing delay, because under real load a long retry chain just makes a person wait longer for what is likely to be the same eventual failure anyway.

**Translations either fully succeed or fully fall back to English, never half and half.** If one piece of an answer fails to translate (for example, a strange character trips up the backup translation service), the entire answer reverts to English rather than showing five sentences in Yoruba and one still in English. A partially translated answer looks more broken and more confusing than a fully English one.

**Two translation engines are used, with one as backup for the other.** Google Translate is tried first for every translation. If it fails, returns an error page, or is rate limited, the app automatically retries using MyMemory Translator instead, so a translation request rarely returns a hard error to the user.

**Translation tries a fast path first, then falls back to a careful one.** A single whole-answer call to Google Translate is attempted first. The result is validated - line count, Markdown table pipe-count per row, and header prefix count must all match the original - before it's trusted. If that check fails, or Google itself errors out, translation falls back to a slower line-by-line and cell-by-cell method, with MyMemory as backup for any piece Google can't handle. Text is also normalized (curly quotes, em-dashes, §, ellipses) before being sent to either service.

**The frontend has no build step.** The entire user interface, layout, styling, and behavior lives inside one static index.html file, using Tailwind CSS (from a CDN) and a small set of self-hosted utility libraries. This removes the need for Node.js, npm, or any bundler on the machine running the app, which keeps the deployed service small and simple to run on a platform like Render.

**Ingestion happens once, separately from the running app.** The six statute PDFs are only ever read and processed by the ingestion script, which is run manually, not automatically. The live app never opens a PDF. It only searches the Pinecone database that ingestion already filled. This keeps the live app's startup fast and keeps a slow, occasional task (reading and chunking PDFs) completely separate from the fast, frequent task (answering a question).

## Tech Stack

### Backend
- Python 3.12
- FastAPI: the web framework that serves the API and the page
- Uvicorn: the server that runs the FastAPI app
- Groq: the AI provider used for query expansion and answer generation (model: openai/gpt-oss-120b)
- Pinecone: the vector database that stores and searches the statute text, using its multilingual-e5-large embedding model
- pypdf: used to pull the text out of the source PDFs during ingestion
- deep-translator: the routing library used to handle translation requests
- Google Translate: the primary translation engine used for multilingual support
- MyMemory: the automatic failover translation engine used when Google Translate is unavailable
- gTTS: used to turn text into spoken audio
- SpeechRecognition, pydub, and imageio-ffmpeg: used together to turn a recorded voice question into text
- slowapi: used to limit how many requests one person can make per minute
- tenacity: used to retry the final AI answer call once if it fails outright
- python-dotenv: used to load settings from the .env file
- uv: the tool used to manage the Python environment and dependencies

### Frontend

- Plain HTML, CSS, and JavaScript, no framework and no build step
- Tailwind CSS (loaded from a CDN link)
- Lucide (icons), Marked (Markdown rendering), and DOMPurify (sanitization) - vendored locally under `static/vendor/`, pinned to specific versions, rather than loaded from a CDN

## Full Project Structure

Every file in this project falls into one of four groups: files the live app needs to run, files used once to set things up, files used occasionally for maintenance, and files used only during development to check the data. Below is every file, what it does, and which group it belongs to.

This matches the exact order the project appears in VS Code's own Explorer panel (folders first, then files, both alphabetical), so you can compare it side by side with your editor and immediately spot anything.

```
CodeAlpha_FAQ_Chatbot/
|
|-- data/
|   |-- docs/                    [Source PDFs, used by ingestion, not by the live app]
|       |-- acja_2015.pdf
|       |-- cama_2020.pdf
|       |-- constitution_1999.pdf
|       |-- criminal_code.pdf
|       |-- evidence_act.pdf
|       |-- penal_code.pdf
|
|-- services/
|   |-- __init__.py              [REQUIRED - empty file, tells Python this folder is a package]
|   |-- audio_handler.py         [REQUIRED - handles voice input and audio output]
|   |-- rag.py                   [REQUIRED - the core AI logic]
|   |-- translator.py            [REQUIRED - handles all translation]
|   |-- vector_store.py          [REQUIRED - talks to Pinecone]
|
|-- static/
|   |-- vendor/                  [Self-hosted JS libraries, pinned versions - not loaded from a CDN]
|       |-- dompurify-3.1.6.js
|       |-- lucide-0.460.0.js
|       |-- marked-12.0.2.js
|   |-- index.html
|   |-- logo.avif
|   |-- logo.png
|   |-- source-links.json
|
|-- cache.py                     [Bounded in-memory response cache for repeated questions]
|-- main.py
|-- measure_data.py             [DEVELOPMENT ONLY - checks how well the section splitting regex is working]
|-- pyproject.toml               [Lists the project's dependencies, used by uv]
|-- README.md                    [This file]
|-- requirements.txt             [Same dependency list, in the plain pip format]
|-- single_pdf.py               [DEVELOPMENT ONLY - quick manual peek at one PDF's extracted text]
|-- uv.lock                      [Locks the exact dependency versions, so every install is identical]
|-- verify_all_sources.py      [DEVELOPMENT ONLY - checks the PDFs will chunk safely before ingesting]
|-- wipe_pinecone.py           [MAINTENANCE - clears the index before a fresh re-ingestion]
```

Not shown above (these are not tracked in Git, so they will not appear in your repository, but you will see them in VS Code locally): the __pycache__ folders, .venv, and .vscode/settings.json.

A closer look at what each one actually does:

### Files the live app needs every time it runs

- **main.py**: This is the front door of the whole app. It starts the FastAPI server, serves the frontend page, and defines the API routes (/api/v1/chat, /api/v1/transcribe, /api/v1/tts, /api/v1/translate). It also checks that the required secret keys are set before it will even start, and keeps a short term memory of each conversation.
- **services/__init__.py**: An empty file. Its only job is to tell Python that the services folder is a package, so files inside it can be imported by main.py.
- **services/rag.py**: This is where the actual thinking happens. It rewrites your question into better search terms, calls the vector database to find relevant statute sections, and sends everything to the Groq AI model to generate a grounded answer.
- **services/vector_store.py**: This file's only job is to talk to Pinecone. It turns your question into a search vector and asks Pinecone which stored statute sections are the closest match.
- **services/translator.py**: Handles every translation in the app. It tries Google Translate first, and switches to MyMemory automatically if Google fails. It is also careful to keep Markdown tables and headers intact during translation, and it protects against showing a mixed language or broken result.
- **services/audio_handler.py**: Converts your recorded voice into text (so the app can understand a spoken question), and converts the AI's text answer into spoken audio you can listen to.
- **static/index.html**: The entire user interface. This one file contains the layout, the styling, and all the JavaScript that talks to the backend. There is no separate frontend build step.
- **static/logo.png / logo.avif**: The app's logo image, shown on the welcome screen and in the header.
- **static/source-links.json**: A small file that maps each law's short name (like CAMA_2020) to a link where the person can read the full original document.
- **static/vendor/**: Self-hosted copies of Lucide, Marked, and DOMPurify, pinned to specific versions, so the frontend doesn't depend on a third-party CDN being reachable at runtime.
- **cache.py**: A small, bounded LRU cache that stores the answer to a session's opening question, so if many different users ask the same common question during a traffic spike, only the first one actually triggers a full search-and-generate cycle - everyone after that gets served from cache instantly. Capped at 500 entries with a 20 minute expiry, so memory use can never grow without bound no matter how much traffic arrives.

### Files used once, to set the project up

- **cloud_index.py**: Run this one time, before you ever ingest data. It creates the Pinecone index itself (the empty container that will later hold all the statute sections). If the index already exists, running it again does nothing harmful, it just skips creation.

### Files used occasionally, for maintenance

- **ingest.py**: Reads every PDF in data/docs/, splits each one into sections, turns each section into a vector, and uploads all of them into Pinecone. You run this the first time to populate the database, and again any time you add a new law or change how the text is split.
- **wipe_pinecone.py**: Deletes everything currently stored in the Pinecone index. You would only run this right before running ingest.py again from scratch, for example after changing the chunking logic and wanting a clean rebuild instead of duplicate old data sitting alongside new data.

### Files used only during development, to check the data

These were used while building and testing the ingestion process. The live app never touches them, and neither does a normal deployment.

- **verify_all_sources.py**: Checks every PDF before ingestion to make sure none of them will produce a chunk so large it would break Pinecone's limits, and flags anything that looks wrong, such as a suspiciously short document.
- **single_pdf.py**: A tiny script used to manually look at the first bit of text extracted from one PDF, just to confirm the PDF reading itself is working correctly.
- **measure_data.py**: Used to check how well the regular expression that detects the start of a new legal section was actually performing against the real PDFs, and to spot any section markers it might be missing.

### Configuration and dependency files

- **.env**: Holds your actual secret API keys. This file is never uploaded to GitHub, it stays only on your own machine, or for a deployed app, is entered directly into Render's settings instead.
- **.gitignore**: A list of files and folders that Git should never track or upload, such as .env, the virtual environment folder, and cached Python files.
- **.python-version**: Tells uv which exact Python version this project expects (3.12).
- **pyproject.toml**: The main list of what packages this project depends on, used by uv to set up the environment.
- **uv.lock**: A locked, exact record of every dependency's precise version. This means anyone who clones the project and runs uv sync gets the exact same setup you had, with no surprises from a package updating in the background.
- **requirements.txt**: The same dependency list as pyproject.toml, but written in the plain format that plain pip understands, for anyone who prefers pip over uv.

## Prerequisites

- Python 3.12 or newer installed and available on your PATH
- Git installed for cloning the repository
- uv installed (recommended, this is how this project was built and is the workflow used below) or pip as a fallback package installer
- A Groq API key and a Pinecone API key
- A modern browser (Chrome or Edge recommended) for full voice dictation and recording support

## Installation and Setup

These steps assume you are using uv, which is how this project was originally set up and is the recommended workflow.

1. Clone the repository

```
git clone https://github.com/big-dominion/CodeAlpha_FAQ_Chatbot.git
cd CodeAlpha_FAQ_Chatbot
```

2. Create the virtual environment

```
uv venv
```

3. Activate the virtual environment

Windows:
```
.venv\Scripts\activate
```

macOS or Linux:
```
source .venv/bin/activate
```

4. Install dependencies

With the virtual environment active, install everything listed in requirements.txt:

```
uv pip install -r requirements.txt
```

If you prefer plain pip instead of uv, the equivalent command is:

```
pip install -r requirements.txt
```

5. Create your .env file

Create a file named .env in the project root and add your own keys. See Environment Variables below for the full list.

## Environment Variables

| Variable | Required | What it does |
|---|---|---|
| PINECONE_API_KEY | Yes | Your Pinecone account key, used to store and search the statute text. |
| PINECONE_INDEX_NAME | No | The name of the Pinecone index to use. Defaults to lexoracle-cloud if not set. |
| GROQ_API_KEY | Yes | Your Groq account key, used to run the AI model. |
| GROQ_MODEL | No | Which Groq model to use. Defaults to openai/gpt-oss-120b if not set. |
| MYMEMORY_EMAIL | No | Registering a real email address with MyMemory raises its free daily translation limit from about 5,000 words to about 50,000 words. Safe to leave unset. |

The app will refuse to start if PINECONE_API_KEY or GROQ_API_KEY is missing. Every other variable has a safe default.

## Running the Application

With the virtual environment active, start the Uvicorn server, which hosts the API and serves the static frontend from the same process:

```
uvicorn main:app --reload
```

Once the server starts, open your browser and navigate to:

```
http://localhost:8000
```

The --reload flag is intended for local development only. Omit it in production.

## Populating the Vector Database

Before the app can answer any question, the six laws need to be loaded into Pinecone. This only needs to be done once, or again later if you add or change a source document.

With the virtual environment active:

1. Create the Pinecone index (only needed the very first time):
   ```
   python cloud_index.py
   ```
2. (Optional but recommended) Check the PDFs are ready:
   ```
   python verify_all_sources.py
   ```
3. Load everything into Pinecone:
   ```
   python ingest.py
   ```

If you ever need to start over completely, clear the index first, then ingest again:

```
python wipe_pinecone.py
python ingest.py
```

## API Reference

All endpoints are prefixed with /api/v1.

### POST /api/v1/chat

Accepts a typed or spoken question. Rate limited to 10 requests per minute per person.

Request (multipart form data):

```
session_id: string (optional, a new one is generated if omitted)
text_input: string (optional if audio_file is provided)
audio_file: file (optional if text_input is provided)
target_lang: string (default "English (US)")
doc_filter: string (default "all", one of: all, constitution, cama, evidence, acja, criminal_code, penal_code)
```

Successful response:

```json
{
  "session_id": "32af336a-7f9b-4c3d-b381-c566591471a0",
  "user_message": "Can a single person register a company?",
  "response_text": "Under Section 815(1), every individual required to be registered must furnish a statement of particulars...",
  "raw_answer": "Under Section 815(1), every individual required to be registered must furnish a statement of particulars...",
  "citations": [
    {
      "key": "CAMA_2020-chunk-1424",
      "doc": "CAMA_2020",
      "section": "Sec 815",
      "match": "84.7%",
      "text": "815. Every individual, firm or company required under this Act to be registered shall..."
    }
  ],
  "history": [
    { "role": "user", "content": "Can a single person register a company?" },
    { "role": "assistant", "content": "Under Section 815(1)..." }
  ]
}
```

Notes:

- raw_answer is always the original English answer, even when target_lang is set to a translated language. response_text is the translated version shown to the user. This lets the tts endpoint translate fresh from clean English later, rather than translating a translation.
- No audio is generated by this endpoint. Voice playback is fetched on demand through /api/v1/tts.

### POST /api/v1/transcribe

Rate limited to 15 requests per minute per person. Converts a recorded voice clip to text only, with no search or answer generation. Called by the frontend before /api/v1/chat when a message is spoken rather than typed, so the person's own voice note and its transcript can appear immediately instead of waiting for the full answer.

Request (multipart form data):

```
audio_file: file (required)
target_lang: string (default "English (US)")
```

Successful response:

```json
{
  "transcript": "Can a single person register a company?"
}
```

Notes:

- If transcription fails (corrupted audio, unsupported format, empty recording), returns a 400 with a `detail` message rather than a generic error.
- The frontend takes the returned transcript and sends it to /api/v1/chat as ordinary `text_input` - this endpoint never triggers a search or an AI answer itself.

### POST /api/v1/tts

Rate limited to 20 requests per minute per person. Converts text into speech and returns base64 encoded MP3 audio, translated into the requested language if needed.

Request body:

```json
{
  "text": "Under Section 815(1), every individual required to be registered must furnish a statement of particulars...",
  "lang": "English (Nigeria)"
}
```

Successful response:

```json
{
  "display_text": "Under Section 815(1), every individual required to be registered must furnish a statement of particulars...",
  "audio_base64": "<base64 encoded mp3 data>",
  "available": true
}
```

Notes:

- display_text is returned alongside the audio so the frontend can update the on-screen answer to match the newly selected language at the same moment the audio changes, keeping the two in sync.
- If the requested language has no voice available in gTTS, available is false and audio_base64 is an empty string, rather than a failed request.

### POST /api/v1/translate

Rate limited to 30 requests per minute per person. Translates text only, with no audio synthesis. Used when the language dropdown changes after an answer is already on screen, so every visible answer can be retranslated immediately without paying the extra cost of a speech request for each one.

Request body:

```json
{
  "text": "Under Section 815(1), every individual required to be registered must furnish a statement of particulars...",
  "lang": "Yoruba"
}
```

Successful response:

```json
{
  "display_text": "Gege bi Abala 815(1) se so, gbogbo eni ti a beere lati forukosile gbodo fi alaye kan ranse..."
}
```

## Known Limitations

- Both the AI model and the translation services run on free tiers with real rate limits. Under unusually heavy simultaneous traffic, a person may occasionally see a message saying the AI service could not be reached, or see an answer come back in English when a translation was requested. Both are the app failing safely and honestly rather than showing a broken or partial result - requests are not retried indefinitely in a way that would make a temporary slowdown worse for everyone else using the app at the same time.
- The app's short term memory (recent conversations, cached translations, cached search expansions, cached first-turn answers) lives only in the server's own memory. It resets whenever the server restarts. This is expected and normal for a single server setup. Moving to something shared, such as Redis, would only be needed if this app ran across multiple worker processes at once.
- Both translation services used (Google Translate and MyMemory) are free, unofficial services accessed without a paid API key. They can occasionally be slow or temporarily unavailable, which is exactly why the app has a backup translator and a fallback to plain English if both struggle.
- Voice recognition quality depends on the browser's recording quality and on Google's speech recognition service. Not every language transcribes with the same accuracy, which is why the transcript is always shown back to the person so a bad recording can be caught before it is answered.
- AI generated answers are deterministic as far as client side settings allow (a fixed temperature and seed are used for every request), but the underlying inference service can still occasionally return slightly different wording. Caching the search expansion step (see Architecture Notes) protects the more important part, which is that the same question always retrieves the same statute sections.
- This tool is a demonstration of grounded AI answering real legal questions with real citations. It is not a substitute for advice from a qualified lawyer.

## Deployment

This project is designed to deploy cleanly as a single web service, for example on Render:

- The Uvicorn process serves both the API routes under /api/v1 and the static frontend from the static directory, in one process, on one port.
- No separate frontend build step, static file host, or Node.js runtime is required.
- Ensure the platform's start command matches the local run command, for example: uvicorn main:app --host 0.0.0.0 --port $PORT. Render provides the port automatically through $PORT, so do not set a fixed port yourself.
- PINECONE_API_KEY and GROQ_API_KEY must be set as environment variables on the deployment platform. The app will refuse to start if either is missing.
- The Pinecone index must already be populated (see Populating the Vector Database) before deploying. Ingestion is a manual, one time (or as needed) process, not something the live app runs on startup.

Once your app is live, come back and add the link at the very top of this file.

## Acknowledgements

Developed by [Samson Kayode Olawumi](https://www.linkedin.com/in/samson-olawumi/) for the CodeAlpha AI Engineering Project.