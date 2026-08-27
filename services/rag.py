"""
DOMINION LexOracle: RAG Orchestrator (Groq Token-Budget Edition)
----------------------------------------------------------------

WHAT THIS FILE DOES (high-level summary):
This module is the "brain" that turns a user's legal question into a
grounded, cited answer. It is the glue between the user, the vector
database, and the Groq-hosted LLM, and it runs a two-step LLM pipeline per
question rather than a single call:

STEP 1 - QUERY EXPANSION (`expand_query`): The user's raw question (which
may be casual, short, or even in a Nigerian language) is sent to the LLM
first, purely to be rewritten into better statutory search keywords. This
expanded string is what actually gets embedded and searched against
Pinecone in `services/vector_store.py` - it is NOT the final answer prompt.

STEP 2 - RETRIEVAL + GENERATION (`query_lex_oracle`): The expanded query
(plus the original raw query as a stability backup - see the comment above
`search_laws` call below) is used to retrieve the most relevant statute
chunks from the vector store. Those chunks are assembled into a
character-budgeted context block, injected into a system prompt with strict
formatting/citation rules, and sent to the LLM a second time to produce the
final answer along with a parallel list of citation cards for the frontend.

Why this file cares so much about determinism:
Legal answers need to be consistent - the same question asked twice should
not "find" a governing section on one run and "not find" it on the next.
Groq's serving infrastructure (continuous batching / MoE routing) makes
LLM outputs non-deterministic even at temperature=0 with a seed set, and
because retrieval is driven entirely by embedding whatever text expansion
produces, even a tiny wording drift in Step 1 can change which chunks Step
2 sees. This file fights that instability on two fronts:
  - `GROQ_SEED` is passed as a best-effort determinism hint to every Groq
    call (see its own comment below for why it's "best effort" and not a
    guarantee).
  - `_expansion_cache` locks in the FIRST successful expansion for any
    given raw question, so repeat questions always embed identically and
    therefore always retrieve identically - this is described in detail
    in the comment above the cache dict itself, since this was found to be
    the actual root cause of inconsistent "partial answer" vs "cannot
    find" behavior for the same question.

Other things this file manages:
- `MAX_CONTEXT_CHARS` / `MAX_ANSWER_TOKENS`: token-budget guardrails so the
  retrieved context plus the generated answer stay within Groq's TPM
  (tokens-per-minute) limits, while still leaving the final answer enough
  room to actually finish a multi-section legal explanation instead of
  being cut off mid-sentence.
- Citation assembly: for every retrieved match, a clean citation dict
  (document, section label, match score, source text) is built for the
  frontend's citation cards, independently of how much of that chunk's
  text actually made it into the LLM's context budget.
- Language handling: no matter what language the user asked in, the system
  prompt forces the final answer to always come back in structured English.

If you're extending this file: `query_lex_oracle(query, doc_filter,
history)` is the single public entry point other modules should call.
`expand_query(user_query)` is an internal helper it calls first.
"""

import os
import logging
from groq import Groq
from services.vector_store import search_laws

logger = logging.getLogger(__name__)
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
# Passed to every Groq call below as a best-effort determinism aid.
# Groq's API is OpenAI-compatible and supports "seed" for this purpose,
# but it is explicitly a best-effort hint, not a hard guarantee - it
# cannot fully override the run-to-run variability that continuous
# batching introduces at the inference-engine level for a
# mixture-of-experts model like this one. Worth trying because it costs
# nothing and can help in some cases, but the same question can still
# occasionally reach a different worded conclusion even with this set,
# if the underlying provider infrastructure changes between requests in
# a way "seed" alone can't compensate for.
GROQ_SEED = 42

# The Groq client used for both LLM calls in this file (query expansion
# and final answer generation). Reads the API key from the environment,
# falling back to a placeholder string if unset - a 40 second timeout is
# set so a hung request fails fast rather than blocking a chat request
# indefinitely.
groq_client = Groq(
    api_key=os.environ.get("GROQ_API_KEY", "your-groq-key"),
    timeout=40.0
)

# Groq model selector (defaults to hosted open-weights model)
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")

# Token budget safety: 14,000 chars (~3,500 tokens) leaves 4,500+ tokens for prompt/generation
MAX_CONTEXT_CHARS = 14000  

# How many tokens the FINAL answer generation call is allowed to produce.
# A structured legal answer with several headings, bullet lists, and
# section citations regularly runs well past 900 tokens once it has to
# cover multiple retrieved provisions - at 900, a genuinely thorough
# answer gets cut off mid-sentence rather than reaching its own
# conclusion, which looked like inconsistent or unreliable behavior
# across runs (a shorter answer finishes cleanly, a longer one on the
# same question gets chopped at a different point each time). Raised to
# give a real multi-section answer room to actually finish.
MAX_ANSWER_TOKENS = 1600

# Caches expand_query's output by the exact raw user question that
# produced it. This exists specifically to defeat run-to-run
# non-determinism: even at temperature=0 with a seed set, Groq's serving
# infrastructure (continuous batching / MoE routing for gpt-oss-120b) can
# still return a slightly different expansion for the identical question
# on different requests. A drifted expansion string embeds differently,
# which can pull a DIFFERENT set of chunks from Pinecone for the same
# question - observed in practice as one run retrieving the Section 20
# chunk and another run not retrieving it at all, which is what actually
# caused "here's a partial answer" vs "I cannot find the answer" for the
# identical question. Caching the expansion means the same question
# always embeds identically and always retrieves identically, which is
# what actually stabilizes the final answer - the seed on the
# answer-generation call alone cannot fix instability introduced
# upstream at retrieval time.
#
# Same caveats as _translation_cache in translator.py: plain in-memory
# dict, not shared across worker processes, cleared on restart. Fine for
# a single-worker dev server. If this app is ever run with multiple
# worker processes or needs to survive restarts, this would need to move
# to something shared like Redis instead.
_expansion_cache = {}


def expand_query(user_query: str) -> str:
    """Translates user input into statutory concepts while preserving core subjects.

    Detailed flow:
    - Check the module-level cache first, keyed by the exact raw
      `user_query` string. If it's a repeat question, return the
      previously-generated expansion immediately with no Groq call at all
      - this is the primary determinism/stability mechanism described in
      the big comment above `_expansion_cache`.
    - Otherwise, build a prompt instructing the model to act as a
      Nigerian legal search expert and rewrite the query into targeted
      statutory search keywords, while explicitly forbidding it from
      inventing section numbers and requiring it to preserve the core
      nouns/subjects of the original question.
    - Call Groq with temperature=0, a generous 300-token ceiling (so the
      model's internal reasoning pass doesn't crowd out the actual
      keyword output), "low" reasoning effort (since this is a simple
      keyword-extraction task, not one needing deep reasoning), and the
      shared GROQ_SEED.
    - If the model returned real (non-empty) content, prepend the
      original user query to it, so the final search string always still
      contains the user's own wording alongside the expansion.
    - If the model returned nothing usable, log a warning and fall back
      to just the original, un-expanded query.
    - Store whatever result was produced in the cache under the original
      `user_query` key, then return it.
    - If the Groq call itself raises (network issue, API error, etc.),
      log a warning and return the original query directly, WITHOUT
      writing anything to the cache - see the comment in the except block
      for why that omission is deliberate.
    """
    if user_query in _expansion_cache:
        return _expansion_cache[user_query]

    expansion_prompt = f"""You are an elite Nigerian legal search expert.
Translate the following user query into a highly targeted search string for a statutory vector database.

CRITICAL RULES:
1. PRESERVE CORE NOUNS: You MUST keep the primary subjects of the query (e.g., "directors", "public company", "minimum").
2. NO HALLUCINATION: Do NOT invent or include specific section numbers (like "section 152").
3. FORMAT: Output ONLY the search keywords as a single string. No intro, no quotes, no markdown.

User Query: {user_query}"""

    try:
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": expansion_prompt}],
            temperature=0.0,
            max_tokens=300,           # more headroom so reasoning tokens don't crowd out actual output
            reasoning_effort="low",   # keep the internal "thinking" pass short for a simple keyword task
            seed=GROQ_SEED
        )
        expanded = response.choices[0].message.content

        if expanded and expanded.strip():
            result = f"{user_query} {expanded.strip()}"
        else:
            logger.warning("Query expansion returned empty content - falling back to original query")
            result = user_query

        _expansion_cache[user_query] = result
        return result

    except Exception as e:
        logger.warning(f"Query expansion failed, falling back to original query. Error: {e}")
        # Deliberately NOT cached: a Groq/network failure is transient by
        # nature, and caching this fallback would permanently lock a
        # question to the degraded un-expanded query for the rest of the
        # process's life, even after Groq recovers.
        return user_query


def query_lex_oracle(query: str, doc_filter: str, history: list):
    """Orchestrates query expansion, vector retrieval, token budgeting, and LLM generation.

    Detailed flow:
    - Step 1: Call `expand_query(query)` to get the statute-search-optimized
      version of the user's question (served from cache on repeat
      questions). Debug-print both the original and expanded query so the
      two can be compared in logs.
    - Step 2: Call `search_laws`, passing the expanded query as the main
      search text AND the original raw `query` as an `extra_queries`
      backup - see the inline comment below for why both are sent
      together rather than just the expanded one.
    - Step 3: Walk through every returned match and, for each one:
        * Pull out its act name, section label, source text, and
          similarity score from its metadata.
        * Build a clean, human-readable section label for the UI (special
          cased for "General Provision", otherwise prefixed with "Sec"
          if not already present).
        * Append a citation dict (key, doc, section, match %, full text)
          to the `citations` list that gets returned to the frontend -
          this list is NOT limited by the token budget below, so every
          retrieved match gets a citation card even if its text had to be
          truncated or skipped from the LLM's actual context.
        * Separately, append as much of that match's text as the running
          MAX_CONTEXT_CHARS token budget allows into `context_text`,
          which IS what gets fed to the LLM: chunks are added in full
          while there's room, one chunk is added truncated (with "...")
          if it's the one that pushes past the budget, and nothing more
          is added once the budget is exhausted.
    - Step 4: Build the system prompt, embedding the assembled
      `context_text` and a strict set of directives: answer ONLY from the
      provided context, give partial answers with an explicit "what's
      missing" note when only part of the question is covered, fall back
      to general rules when an exact scenario isn't stated, and refuse
      with a fixed sentence if the context is completely irrelevant. It
      also mandates Markdown formatting (bold, bullets, and - notably -
      REQUIRED tables for any comparable/structured data), and forces the
      final answer to always be in English regardless of what language
      the user asked in.
    - Step 5: Assemble the message list: the system prompt first, then up
      to the last 4 messages of conversation `history` (for short-term
      context), then the current raw `query` as the newest user message.
    - Step 6: Call Groq for the final answer with temperature=0,
      `MAX_ANSWER_TOKENS` as the output ceiling, and the shared
      `GROQ_SEED`. Print the response's `system_fingerprint` purely as a
      diagnostic (see its own comment below for how to use it). If a
      real answer came back, normalize em-dashes and en-dashes to a
      plain " - " for consistent rendering.
    - Step 7: If the Groq call itself raises, log the full exception and
      fall back to a fixed "System Error" message instead of letting the
      exception propagate up to the caller.
    - Return a dict with the raw answer text and the full citations list,
      for the calling code (e.g. the Flask/FastAPI route) to render.
    """
    
    legal_query = expand_query(query)
    logger.debug(f"Original query: {query!r}")
    logger.debug(f"Expanded query sent to Pinecone: {legal_query!r}")

    # The raw, un-expanded question is passed alongside the expanded one
    # as a stable anchor for retrieval - see search_laws' docstring in
    # vector_store.py for why: the expansion step's own wording can drift
    # slightly between otherwise-identical requests, and retrieval is
    # driven entirely by whatever text gets embedded, so a decisive
    # provision could get missed on a run where expansion happened to
    # drift away from it. Querying on both and merging means that
    # provision only needs to be found by ONE of the two phrasings.
    # expand_query is now cached above, so this pairing is now mostly a
    # second line of defense rather than the primary stabilizer - it
    # still protects against a genuinely first-time expansion missing a
    # decisive provision, before that expansion gets locked in by cache.
    matches = search_laws(legal_query, doc_filter, extra_queries=[query])

    context_text = ""
    citations = []
    accumulated_chars = 0

    for match in matches:
        metadata = match.get("metadata", {})
        act = metadata.get("act_name", "Unknown Act")
        section = metadata.get("section", "Unknown Section")
        text = metadata.get("text", "")
        score = round(match.get("score", 0.0) * 100, 1)

        vector_id = match.get("id", "")

        # Clean display label for UI cards
        if section == "General Provision":
            section_display = section
        else:
            section_display = section if section.startswith("Sec") else f"Sec {section}"

        # Preserve full citations list for the frontend cards
        citations.append({
            "key": vector_id,
            "doc": act,
            "section": section_display,
            "match": f"{score}%",
            "text": text
        })

        # Token Budgeting: Append text to context payload up to character threshold
        if accumulated_chars + len(text) <= MAX_CONTEXT_CHARS:
            context_text += f"ACT: {act} | SECTION: {section}\nTEXT: {text}\n\n"
            accumulated_chars += len(text)
        elif accumulated_chars < MAX_CONTEXT_CHARS:
            remaining_budget = MAX_CONTEXT_CHARS - accumulated_chars
            context_text += f"ACT: {act} | SECTION: {section}\nTEXT: {text[:remaining_budget]}...\n\n"
            accumulated_chars += remaining_budget

    system_prompt = f"""You are DOMINION LexOracle, an elite statutory legal intelligence engine.

CONTEXT PROVIDED:
{context_text}

YOUR DIRECTIVE:
1. Read the provided CONTEXT.
2. Answer the user's question using ONLY the provided CONTEXT.
3. If the context contains the answer to ONLY PART of the question, give the partial answer and explicitly state what is missing.
4. If the exact specific scenario is not explicitly stated, but a general rule is provided, state the general rule and clarify the distinction based on the text.
5. ONLY if the context is completely irrelevant, reply with "I cannot find the answer in the provided statutes." Do not use outside knowledge.

FORMATTING:
- Cite the specific sections (e.g., "Under Section 271..." or "According to Section 275...") based on the text.
- Use Markdown formatting to make the response highly readable.
- Use **bolding** for emphasis and key terms.
- Use bullet points to break down complex lists, requirements, or conditions so they are easily scannable and not jumbled into a single paragraph.
- MANDATORY: If the answer compares two or more items, lists conditions with corresponding outcomes, or presents any data with two or more comparable columns, you MUST format it as a Markdown table. Do not present comparable structured data as plain prose or bullets when a table applies. This is not optional based on your judgment - if the content fits a table, use one.

CRITICAL LANGUAGE INSTRUCTION:
Regardless of the language used in the user's prompt (whether Yoruba, Hausa, Igbo, French, etc.), you MUST ALWAYS generate your complete answer in structured, clear English. 
"""

    messages = [{"role": "system", "content": system_prompt}]

    for msg in history[-4:]:
        messages.append({"role": msg["role"], "content": msg["content"]})

    messages.append({"role": "user", "content": query})

    try:
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=0.0,
            max_tokens=MAX_ANSWER_TOKENS,
            seed=GROQ_SEED
        )
        raw_answer = response.choices[0].message.content

        # Diagnostic only, not used for any logic: if the exact same
        # question ever produces two different conclusions again even
        # with GROQ_SEED set, compare this value between the two runs'
        # logs. A different fingerprint confirms the underlying model
        # serving infrastructure genuinely changed between requests -
        # something no client-side parameter can compensate for - rather
        # than the seed simply not having taken effect.
        logger.debug(f"Groq system_fingerprint: {getattr(response, 'system_fingerprint', None)!r}")

        if raw_answer:
            raw_answer = raw_answer.replace('\u2014', ' - ').replace('\u2013', ' - ')
    except Exception as e:
        logger.exception(f"Groq API Error: {e}")
        raw_answer = "System Error: Unable to reach the Groq inference engine."

    return {
        "raw_answer": raw_answer,
        "citations": citations
    }