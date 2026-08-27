"""
DOMINION LexOracle: Vector Search Client
----------------------------------------

WHAT THIS FILE DOES (high-level summary):
This module is the retrieval half of the RAG pipeline - it's responsible
for turning a search query string into a ranked list of relevant statute
chunks pulled from Pinecone, the vector database this app uses to store
embedded law text. The single public entry point other modules should call
is `search_laws(query, doc_filter, top_k, extra_queries)`.

The core mechanics, per query string:
1. EMBED: The query text is sent to Pinecone's own hosted inference
   endpoint (`pc.inference.embed`, using the "multilingual-e5-large" model)
   to turn it into a numeric vector - this is a REMOTE call, not a local
   model, hence "Cloud Inference" in the module docstring.
2. QUERY: That vector is used to search the configured Pinecone `index`
   for its nearest neighbors, optionally restricted to a specific act via
   `filter_dict` (built from `doc_filter`), returning the `top_k` closest
   matches with their metadata attached.

Both of those steps are wrapped up in the internal `_embed_and_query`
helper, which runs exactly once per distinct phrasing of a question.

Why `search_laws` can run MORE than one embed+query pass per call
(the `extra_queries` mechanism):
The RAG orchestrator's LLM-based query expansion step (`expand_query` in
rag.py) is not perfectly reproducible request-to-request, even at
temperature 0 - Groq's continuous-batched inference can return a slightly
different completion for what looks like an identical prompt, depending on
what else is being processed in the same batch at the time. Since
retrieval here is driven entirely by whatever exact text gets embedded, a
small wording drift in that expansion can shift which chunks land in the
top-K purely by chance, occasionally dropping a genuinely decisive
provision that a slightly different phrasing of the SAME underlying
question would have retrieved. `search_laws` hedges against this by
letting the caller pass one or more `extra_queries` (rag.py passes the
user's raw, un-expanded question as one such extra) - each distinct
phrasing is embedded and queried separately, and the results are merged
by chunk ID, keeping each chunk's BEST score across every phrasing it
appeared under. That way a chunk only needs to clear the score threshold
under ANY ONE of the phrasings to survive, not every one of them.

This comes at a real cost: each extra phrasing is a full additional
embedding call plus Pinecone query, so passing N phrasings costs roughly
N times the latency/expense of a single search. `search_laws` de-duplicates
identical phrasings automatically (e.g. if query expansion happened to
fail and fall back to the raw query verbatim, it won't be embedded twice),
but distinct phrasings are always run separately and merged.

After all phrasings have been queried and merged, `search_laws` applies
`SCORE_THRESHOLD` to drop weak matches, sorts what's left by score
descending, and returns at most `top_k` results.

If you're extending this file: `search_laws(...)` is the only function
other modules should call. `_embed_and_query(...)` is an internal helper
used by it to run a single phrasing's embed-then-query pass.
"""

import os
import logging
from pinecone import Pinecone
from dotenv import load_dotenv

# Loaded here as well as in main.py so this module still resolves its
# environment correctly if it's ever imported or run standalone (e.g. a
# quick script or REPL session that doesn't go through main.py's own
# load_dotenv() call first). When main.py IS the entry point, its
# load_dotenv(override=True) already ran first, so this call is a no-op
# in that case - python-dotenv does not clear or override values that
# are already set in the environment unless override=True is passed,
# and it isn't here.
load_dotenv()

logger = logging.getLogger(__name__)

# The Pinecone client used for both embedding (pc.inference.embed) and
# indexing (pc.Index below). Reads the API key from the environment,
# falling back to a placeholder string if unset.
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY", "your-pinecone-key"))

# Reads the index name from .env instead of hardcoding it, so changing
# PINECONE_INDEX_NAME actually takes effect instead of silently doing
# nothing. Falls back to "lexoracle-cloud" - the name this app has
# always used - if the env var is ever unset, so nothing breaks for an
# existing local setup or a deploy that forgot to set it explicitly.
INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "lexoracle-cloud")

# The actual Pinecone index handle every query in this module runs
# against, resolved once at import time using INDEX_NAME above.
index = pc.Index(INDEX_NAME)

# Minimum similarity score (Pinecone's cosine-similarity-based score, on
# roughly a 0-1 scale) a match must reach to be considered relevant
# enough to keep. Matches below this are treated as noise and dropped in
# search_laws before the top_k cut is applied.
SCORE_THRESHOLD = 0.4


def _embed_and_query(query: str, filter_dict: dict, top_k: int) -> list:
    """
    Embeds a single query string and runs one Pinecone lookup for it.
    Split out from search_laws so that function can call this once per
    phrasing of the same question and merge the results, instead of
    being limited to whatever one embedding happened to retrieve.

    Detailed flow:
    - Call Pinecone's hosted inference endpoint to embed `query` using
      the "multilingual-e5-large" model, explicitly marking it as a
      "query"-type input (as opposed to a "passage"/document-type input,
      which the e5 model family embeds differently) via the
      `input_type` parameter.
    - Pull the resulting vector out of the response (`res[0].values`) -
      there's only one input, so only one embedding comes back.
    - Query the configured Pinecone `index` with that vector, applying
      `filter_dict` (e.g. restricting to one specific act) and asking
      for the top `top_k` nearest neighbors, with `include_metadata=True`
      so each match comes back with its act name, section, and text
      attached.
    - Return just the list of matches from the response (an empty list
      if the response has no "matches" key at all).
    """
    res = pc.inference.embed(
        model="multilingual-e5-large",
        inputs=[query],
        parameters={"input_type": "query"}
    )
    query_vector = res[0].values

    search_response = index.query(
        vector=query_vector,
        filter=filter_dict,
        top_k=top_k,
        include_metadata=True
    )
    return search_response.get("matches", [])


def search_laws(query: str, doc_filter: str = "all", top_k: int = 10, extra_queries: list = None):
    """
    Embeds the search query remotely and queries Pinecone for the top
    relevant chunks. If extra_queries is given, each of those is also
    embedded and queried, and every result is merged into one pool
    before the score threshold and top_k cut are applied.

    Why extra_queries exists: rag.py's LLM-based query expansion is not
    perfectly reproducible request-to-request even at temperature 0 -
    Groq's continuous-batched inference can return slightly different
    token completions for an identical prompt depending on what else is
    being processed in the same batch, which is a known characteristic
    of that kind of high-throughput serving rather than a bug in the
    prompt itself. That means the exact wording handed to embedding can
    drift slightly between two otherwise-identical questions, and
    because retrieval here is driven entirely by that one embedding, the
    drift can shift which chunks land in the top-K purely by chance -
    occasionally dropping a decisive provision that a slightly different
    phrasing of the very same question would have found.

    Querying with more than one phrasing and merging results (keeping
    each chunk's best score across every phrasing it was found under)
    hedges against that: a chunk only needs to score above threshold
    under ANY one of the phrasings to survive into the merged results,
    not every one of them. rag.py uses this by passing the user's raw,
    un-expanded question as an extra_queries entry alongside the LLM's
    expanded version, so the raw question acts as a stable anchor that
    doesn't depend on the expansion step succeeding or landing the same
    way twice.

    The trade-off: this issues one embedding call and one Pinecone query
    per phrasing instead of one overall, so passing extra_queries roughly
    doubles (or more) the embedding/Pinecone cost and latency of a single
    search_laws call. Only pass phrasings that are actually meaningfully
    different from each other - an identical duplicate is skipped
    automatically below rather than run twice.

    Detailed flow:
    - Everything below is wrapped in a single try/except: any failure
      anywhere in embedding, querying, or merging is logged with the
      original query and filter for context, and results in an empty
      list being returned rather than an exception propagating up to
      the caller (rag.py) and breaking the whole chat request.
    - Build `filter_dict`: empty (no restriction) if `doc_filter` is the
      default "all", otherwise an exact-match filter on `act_name`
      (upper-cased, since act names are presumably stored upper-cased in
      Pinecone's metadata).
    - Build the de-duplicated list of phrasings to actually run: start
      with `query` itself, then append each entry from `extra_queries`
      (defaulting to an empty list if None was passed) - but only if
      it's truthy AND not already seen, so an extra_queries entry that
      happens to be identical to the main query (e.g. because query
      expansion failed and fell back to the raw query verbatim) doesn't
      trigger a second, redundant embed+query call.
    - For every phrasing in that de-duplicated list, call
      `_embed_and_query` and merge its matches into `merged_by_id`,
      keyed by each match's Pinecone ID:
        * If this chunk ID hasn't been seen yet, store it.
        * If it HAS been seen already (found under an earlier phrasing
          too), only overwrite the stored entry if this phrasing's score
          for it is HIGHER than the previously stored score - so the
          chunk's best score across all phrasings wins, never whichever
          phrasing happened to run last.
    - Take all the merged matches, sort them by score descending, filter
      out anything below `SCORE_THRESHOLD`, and return at most the first
      `top_k` of what's left.
    """
    try:
        filter_dict = {}
        if doc_filter != "all":
            filter_dict = {"act_name": {"$eq": doc_filter.upper()}}

        # De-duplicate so an extra_queries entry identical to the main
        # query (e.g. expansion failed and fell back to the raw query
        # verbatim) doesn't cost a second, redundant embed+query call.
        seen = set()
        queries_to_run = []
        for q in [query] + list(extra_queries or []):
            if q and q not in seen:
                seen.add(q)
                queries_to_run.append(q)

        merged_by_id = {}
        for q in queries_to_run:
            for match in _embed_and_query(q, filter_dict, top_k):
                match_id = match.get("id")
                score = match.get("score", 0.0)
                # Keep this chunk's best score across every phrasing it
                # showed up under, not just whichever phrasing happened
                # to be queried last.
                if match_id not in merged_by_id or score > merged_by_id[match_id].get("score", 0.0):
                    merged_by_id[match_id] = match

        all_matches = sorted(merged_by_id.values(), key=lambda m: m.get("score", 0.0), reverse=True)
        filtered = [m for m in all_matches if m.get("score", 0.0) >= SCORE_THRESHOLD]
        return filtered[:top_k]

    except Exception as e:
        logger.exception(f"Vector Store Error for query={query!r} filter={doc_filter!r}: {e}")
        return []