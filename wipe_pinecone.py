"""
DOMINION LexOracle: Pinecone Index Wipe Utility
-----------------------------------------------

WHAT THIS FILE DOES (high-level summary):
This is a small, standalone, manually-run destructive utility script - like
`cloud_index.py` and `ingest.py`, it is NOT imported or called by the
running app itself, and it is meant to be run directly from the command
line by a person, deliberately, when they specifically want to clear out
existing data.

Its one job is to delete every vector currently stored in the
"lexoracle-cloud" Pinecone index's default namespace, leaving the index
itself intact (still created, still configured with the right dimension
and metric - see `cloud_index.py`) but completely empty of statute chunks.
This is normally run as a deliberate first step before re-running
`ingest.py` from scratch - for example, after fixing a bug in
`chunk_text_by_section` or `extract_section_label` in ingest.py, where
simply re-running ingestion on top of the old, differently-chunked data
could leave a confusing mix of old and new chunk boundaries/IDs behind
instead of a clean, consistent dataset.

Why this is dangerous and worth calling out clearly: `index.delete(delete_all=True)`
is irreversible from this app's side - there's no confirmation prompt, no
dry-run mode, and no undo. Running this script wipes ALL ingested statute
data immediately, for every Act, not just one. Whoever runs it needs to be
sure they actually intend to re-ingest everything afterward, since between
this script finishing and `ingest.py` finishing again, `services/vector_store.py`
would find nothing at all for any question asked against the live app.

This script does NOT delete or recreate the Pinecone index itself (that's
`cloud_index.py`'s job) - it only empties the vectors inside an index that
is assumed to already exist. Like `cloud_index.py` and `ingest.py`, it's
guarded by `if __name__ == "__main__":`, so importing this module
elsewhere would never trigger a wipe as a side effect - only running it
directly, or calling `wipe_index()` explicitly, does that.
"""

import os
from pinecone import Pinecone
from dotenv import load_dotenv

# Loads variables from a local .env file (e.g. PINECONE_API_KEY) into the
# process environment before anything below tries to read them.
load_dotenv()

# The Pinecone client used to connect to and wipe the index below. Reads
# the API key from the environment, falling back to a placeholder string
# if unset (matching the same fallback style used in services/vector_store.py).
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY", "your-pinecone-key"))

# The exact index name this script wipes. Hardcoded here rather than read
# from PINECONE_INDEX_NAME in the environment (unlike services/vector_store.py
# and ingest.py, which both read that env var with this same string as
# their fallback) - if the running app is ever pointed at a different
# index via that env var, this script would still need to be updated to
# match, or it would wipe the wrong (or a nonexistent) index.
INDEX_NAME = "lexoracle-cloud"

def wipe_index():
    """
    Connects to the "lexoracle-cloud" Pinecone index and deletes every
    vector in its default namespace, leaving the index itself in place
    but completely empty.

    Detailed flow:
    - Print a status message announcing which index is about to be
      connected to.
    - Get a handle to that index via `pc.Index(INDEX_NAME)`.
    - Call `index.delete(delete_all=True)` - this removes every single
      vector currently stored in the index's default namespace in one
      call. There is no batching, no confirmation step, and no way to
      target only a subset (e.g. just one Act) with this specific call -
      it is all-or-nothing for the default namespace.
    - Print a final confirmation message once the delete call returns,
      signaling the index is now empty and ready for a fresh
      `ingest.py` run.
    """
    print(f"Connecting to Pinecone index: {INDEX_NAME}...")
    index = pc.Index(INDEX_NAME)
    
    # This deletes all vectors in the default namespace
    index.delete(delete_all=True)
    print("All old chunks have been wiped clean! You are ready to re-ingest.")

# Only runs wipe_index() when this file is executed directly (e.g.
# `python wipe_pinecone.py`), not when it's imported as a module elsewhere
# in the app - so simply importing this file never has the destructive
# side effect of deleting Pinecone data on its own.
if __name__ == "__main__":
    wipe_index()