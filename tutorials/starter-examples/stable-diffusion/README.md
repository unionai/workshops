# Stable Diffusion Image Generation

Generate images from text prompts using SDXL Turbo on Flyte.

## What it does

- Loads the SDXL Turbo model from HuggingFace (no auth required)
- Generates an image from a text prompt using the diffusers pipeline
- Displays the result in a Flyte report with the prompt and generated image
- Returns the image as a `flyte.io.File`

## Setup

```bash
cd tutorials/starter-examples/stable-diffusion

uv venv .venv --python 3.11
source .venv/bin/activate

uv pip install -r requirements.txt
```

## Flyte Cluster (for remote runs)

To run remotely, configure your Flyte cluster endpoint:

```bash
flyte create config \
    --endpoint <your-endpoint> \
    --auth-type headless \
    --builder remote \
    --domain development \
    --project flytesnacks
```

Don't have a cluster? Request access at [flyte.org](https://flyte.org/).

## Run

**Remote:**
```bash
uv run flyte run stable_diffusion.py generate --prompt "a cat astronaut floating in space, digital art"
```

**Local:**
```bash
uv run flyte run --local stable_diffusion.py generate --prompt "a cat astronaut floating in space, digital art"
```

## Notes

- Uses GPU with CUDA 12.4 PyTorch for fast generation
- SDXL Turbo is optimized for few steps — try `--steps 4` for faster results