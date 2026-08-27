"""
DOMINION LexOracle: Pinecone Cloud Index Provisioner
----------------------------------------------------

WHAT THIS FILE DOES (high-level summary):
This is a one-off setup/provisioning script, not something imported and
called by the running app during normal request handling (contrast with
`services/vector_store.py`, which queries an already-existing index at
request time). Its only job is to make sure the Pinecone serverless index
this whole app depends on actually exists before anything tries to write
to or query it - it's meant to be run manually (`python cloud_index.py`,
or whatever this file is named) once when setting up a new environment, or
again after switching to a fresh Pinecone project/account.

What "provisioning" means here, concretely:
- It connects to Pinecone using the API key from the environment.
- It checks whether an index named "lexoracle-cloud" already exists in
  that Pinecone account.
- If it does NOT exist yet, it creates one with the exact configuration
  the rest of the app expects: 1024 dimensions (this must match whatever
  embedding model produces the vectors being stored - here,
  "multilingual-e5-large", the same model `services/vector_store.py` uses
  to embed queries), cosine similarity as the distance metric, and a
  serverless spec hosted on AWS in the us-east-1 region.
- Since index creation on Pinecone's side is asynchronous (the request
  returns before the index is actually usable), it then polls
  `describe_index` in a loop, checking the `status['ready']` flag once
  per second, until Pinecone reports the index is actually live.
- If an index with this name already exists, it does nothing further and
  just prints a message confirming that - this makes the script safe to
  re-run any number of times without erroring or trying to recreate an
  index that's already there.

Why this matters to keep in sync with the rest of the app: if this
script's `dimension` (1024) ever stops matching the actual output size of
whatever embedding model `services/vector_store.py` uses, or if
`INDEX_NAME` here ever drifts out of sync with `PINECONE_INDEX_NAME` /
the default in vector_store.py, queries against the index would fail or
behave unexpectedly. This file and vector_store.py are two separate
places that both need to agree on the same index name and vector shape.

This script is guarded by the `if __name__ == "__main__":` block at the
bottom, so importing this module elsewhere (rather than running it
directly) does NOT automatically trigger index creation - only an
explicit `setup_index()` call, or running this file directly, does that.
"""

import os
import time
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

# Loads variables from a local .env file (e.g. PINECONE_API_KEY) into the
# process environment before anything below tries to read them.
load_dotenv()

# Initialize Pinecone client, reading the API key from the environment.
# Unlike services/vector_store.py, this does not fall back to a
# placeholder string if the key is unset - if PINECONE_API_KEY is
# missing, this will simply fail once it actually tries to talk to
# Pinecone, which is acceptable for a manually-run setup script.
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

# The exact index name this script provisions. This is hardcoded here
# (unlike services/vector_store.py, which reads PINECONE_INDEX_NAME from
# the environment with this same string as its fallback default) - if
# you ever change PINECONE_INDEX_NAME in .env to point the running app
# at a different index, remember this script would still create/check
# "lexoracle-cloud" specifically unless it's updated to match.
INDEX_NAME = "lexoracle-cloud"

def setup_index():
    """
    Ensures the "lexoracle-cloud" Pinecone serverless index exists and is
    ready to use, creating it if necessary.

    Detailed flow:
    - Fetch the list of all indexes currently in this Pinecone account
      and pull out just their names.
    - If INDEX_NAME is NOT among them:
        * Print a status message and issue the create_index request,
          specifying:
            - dimension=1024, which MUST match the output size of the
              "multilingual-e5-large" embedding model used elsewhere in
              this app (services/vector_store.py) - a mismatch here would
              make every future query against this index fail.
            - metric="cosine" for similarity scoring.
            - a ServerlessSpec targeting AWS's us-east-1 region, meaning
              Pinecone manages the underlying infrastructure/scaling
              rather than this app running a dedicated pod.
        * Pinecone's index creation is asynchronous - the create_index
          call returns before the index is actually usable - so this
          then loops, calling describe_index(INDEX_NAME) and checking
          its status['ready'] flag, sleeping 1 second between checks,
          until that flag finally comes back True.
        * Once ready, print a final confirmation message.
    - If INDEX_NAME already exists, skip all of the above and just print
      a message noting it already exists - this makes the function (and
      the script as a whole) idempotent and safe to run repeatedly.
    """
    existing_indexes = [idx.name for idx in pc.list_indexes()]

    if INDEX_NAME not in existing_indexes:
        print(f"Creating new 1024-dimension index '{INDEX_NAME}'...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=1024,  # Required for multilingual-e5-large
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        print("Index creation request submitted. Waiting for readiness...")
        while not pc.describe_index(INDEX_NAME).status['ready']:
            time.sleep(1)
        print(f"Index '{INDEX_NAME}' is live and ready for ingestion!")
    else:
        print(f"Index '{INDEX_NAME}' already exists. Skipping creation.")

# Only runs setup_index() when this file is executed directly (e.g.
# `python cloud_index.py`), not when it's imported as a module elsewhere
# in the app - so simply importing this file never has the side effect
# of trying to create a Pinecone index.
if __name__ == "__main__":
    setup_index()