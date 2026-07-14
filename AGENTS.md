# Workshops

Production-grade AI tutorials and workshop content for Union.ai / Flyte.
Part of the DevRel workspace. See `$GITREPOS/workspace/workspace.md` for the full constellation.

## What this repo is

Hands-on tutorials that show how to build real AI workflows with Flyte 2. Each tutorial targets a specific domain (fraud detection, fine-tuning, agents, etc.) and is designed to work both locally and on Union's managed platform.

## Structure

```
tutorials/
  fraud-detection-feast/   # Feast + Flyte fraud detection pipeline
  ...                      # Each tutorial in its own directory
docs/                      # Blog posts and written guides
```

## Build and test

```bash
# Environment setup (per tutorial)
cd tutorials/<tutorial-name>
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt

# Local execution
flyte run --local <workflow_file.py> <workflow_name>

# Remote execution
flyte run <workflow_file.py> <workflow_name>

# Deploy apps
flyte deploy <app_file.py> <env_name>
```

## Key conventions

- `uv` for environment management, not pip or conda
- Each tutorial has its own `requirements.txt`
- Use `.with_pip_packages()` for remote image builds, not `.with_requirements()`.
  `.with_requirements()` stores a **relative path** and re-resolves it at runtime, so it breaks
  anywhere the working directory isn't yours and the file isn't in the code bundle — most
  notably when an app pod launches a task (`[Errno 2] No such file or directory:
  'requirements.txt'`). Keep `requirements.txt` for the local venv, and keep the two in sync.
- Flyte 2 SDK only: `import flyte`, `@env.task`
- Test locally before remote deployment
- Each tutorial includes a README with setup, steps, and example commands
- Check `$GITREPOS/flyte-sdk/examples/` for official patterns before guessing API

## Flyte SDK reference

When building tutorials or debugging Flyte behavior, check these for patterns:

- SDK source: `$GITREPOS/flyte-sdk/src/`
- SDK examples: `$GITREPOS/flyte-sdk/examples/`
- Union docs: `$GITREPOS/unionai-docs/content/`

Key example directories:
- `examples/apps/` - FastAPI, Gradio, Streamlit deploy patterns
- `examples/basics/` - Core concepts (tasks, files, types, resources)
- `examples/genai/` - AI/agent examples
- `examples/ml/` - Training, serving, model artifacts

## Out of scope

- Do not modify files outside the `tutorials/` or `docs/` directories without asking
- Do not change the top-level README table structure without asking
- Do not add dependencies to a tutorial's requirements.txt without confirming they're needed

## Known gotchas

- `flyte deploy` pickles the `app` object. Unpicklable state (queues, threads) must be set up in `@env.on_startup`.
- `include` paths are relative to the app script's directory.
- MPS (Apple Silicon) crashes on models >135M params. Default to CPU for local dev.
