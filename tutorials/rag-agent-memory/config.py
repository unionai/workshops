"""Task environments: image, compute, and secrets.

Two environments, because the first half of this tutorial has no model in it:

    index_env — steps 0, 1, 3. Embeddings only, and those run locally on CPU.
                No API key, no secret, nothing to register.
    llm_env   — steps 2, 4, 5. These call Claude, so they need the key.

That split is the point, not an accident. You can build an index and search it
before you have found your Anthropic key, which means step 1 works for everyone
in the room on the first try.

Both environments share one image, so the first run builds it and the rest
start warm.
"""

from dotenv import load_dotenv

import flyte

load_dotenv()  # ANTHROPIC_API_KEY for `flyte run --local`; ignored on the cluster

# Packages are listed inline rather than via .with_requirements("requirements.txt"):
# that stores a *relative path* and re-resolves it at runtime, which breaks inside
# a pod (cwd is not this directory, and the file is not in the code bundle) — most
# visibly when step 5's app pod launches a task. Keep this list and
# requirements.txt in sync; requirements.txt is for your local venv.
image = flyte.Image.from_debian_base(name="rag-agent-memory").with_pip_packages(
    "chromadb>=0.5.0",
    "sentence-transformers>=3.0.0",
    "qdrant-client>=1.9",  # --store qdrant, embedded mode
    "anthropic>=0.40.0",
    "python-dotenv",
    "datasets>=3.0.0",  # step 0's --source hf
    "umap-learn>=0.5.5",  # step 3
    "plotly>=5.20.0",
    "numpy",
    "gradio>=6.0",  # step 5; v6 removed Chatbot's `type` argument
    "openai>=1.50.0",  # llm.py's OpenAI-compatible path
)

# Steps 0 and 1 — build the index and search it. No model, no secret.
index_env = flyte.TaskEnvironment(
    name="rag-index",
    image=image,
    # Embedding is the heavy part and bge-small is CPU-friendly; 2 cores chews
    # through the default corpus in well under a minute.
    resources=flyte.Resources(cpu=2, memory="4Gi"),
)

# Steps 2, 4, 5 — anything that calls a model. The secret must live in the
# project/domain you run in:
#   flyte create secret ANTHROPIC_API_KEY -p flytesnacks -d development
llm_env = flyte.TaskEnvironment(
    name="rag-agent-memory",
    image=image,
    resources=flyte.Resources(cpu=2, memory="4Gi"),
    secrets=[flyte.Secret(key="ANTHROPIC_API_KEY", as_env_var="ANTHROPIC_API_KEY")],
    # Step 2 lives here but calls `index()`, which lives in `index_env`. Locally
    # that just works — one process, both environments imported. On a cluster it
    # does not: only environments reachable from the one you launched get
    # deployed, so the call fails at runtime with
    #     Environment 'rag-index' not found in image cache.
    # Declaring the dependency makes `rag-index` deploy alongside this one.
    depends_on=[index_env],
)
