import base64
import torch

import flyte
import flyte.io
import flyte.report

env = flyte.TaskEnvironment(
    name="stable-diffusion",
    image=(
        flyte.Image.from_debian_base()
        .with_commands(
            [
                "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124"
            ]
        )
        .with_requirements("requirements.txt")
    ),
    resources=flyte.Resources(cpu=2, memory="8Gi", gpu=1),
)


@env.task(report=True)
async def generate(prompt: str, steps: int = 30) -> flyte.io.File:
    """Generate an image from a text prompt using Stable Diffusion."""
    from diffusers import AutoPipelineForText2Image

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sdxl-turbo",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        variant="fp16" if device == "cuda" else None,
    ).to(device)

    image = pipe(prompt, num_inference_steps=steps, guidance_scale=0.0).images[0]

    path = "/tmp/output.png"
    image.save(path)

    with open(path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()
    await flyte.report.replace.aio(
        f"<h2>Stable Diffusion</h2>"
        f"<p><b>Prompt:</b> {prompt}</p>"
        f'<img src="data:image/png;base64,{img_b64}" style="max-width:512px" />'
    )
    await flyte.report.flush.aio()

    return await flyte.io.File.from_local(path)


# uv run flyte run stable_diffusion.py generate --prompt "a cat astronaut floating in space, digital art"
