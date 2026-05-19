# Future Fine-Tuning Tutorials

Ideas for follow-up tutorials using this pipeline as a template.

## Next Models to Try

| Model | Params | Why | LoRA targets |
|-------|--------|-----|-------------|
| Qwen/Qwen2.5-0.5B | 500M | Drop-in swap, big quality jump over SmolLM2 | Same as default |
| Qwen/Qwen3-0.6B | 600M | Built-in thinking/reasoning | Same as default |
| google/gemma-4-1b | 1B | Latest from Google, multimodal-ready | Check modules |
| google/gemma-3-1b-pt | 1B | Strong small model from Google | Check modules |
| meta-llama/Llama-3.2-1B | 1B | Most popular family, gated (shows HF_TOKEN) | Same as default |
| microsoft/Phi-4-mini | 3.8B | Punches above its weight, QLoRA actually needed on T4 | `q_proj`, `k_proj`, `v_proj`, `dense` |
| Qwen/Qwen2.5-1.5B | 1.5B | Where QLoRA starts to matter | Same as default |

## Datasets to Pair

| Dataset | Task | Good pairing |
|---------|------|-------------|
| `tatsu-lab/alpaca` | General instruction following | Any model, classic benchmark |
| `sahil2801/CodeAlpaca-20k` | Code generation | Qwen2.5-0.5B — "SQL but for Python" |
| `FinGPT/fingpt-sentiment-train` | Financial sentiment | Phi-4-mini — real business domain, QLoRA needed |
| `medalpaca/medical_meadow_medical_flashcards` | Medical Q&A | Gemma-4 — domain adaptation, biotech angle |
| `OpenAssistant/oasst2` | Chat/dialogue | Multi-turn chatbot fine-tuning |
| `iamtarun/python_code_instructions_18k` | Python from docstrings | Clean format, easy to evaluate |
| `gsm8k` | Math reasoning | Shows reasoning improvements |
| `argilla/distilabel-capybara-dpo-7k-binarized` | DPO/preference | Different training paradigm |

## Suggested Workshop Series

1. **This tutorial** — SmolLM2 + SQL (LoRA basics, all three methods compared)
2. **Qwen2.5-0.5B + CodeAlpaca** — code generation, same pipeline template
3. **Phi-4-mini + FinGPT sentiment** — QLoRA actually needed, real business domain
4. **Gemma-4 + medical flashcards** — domain adaptation, biotech angle

Each reuses the same pipeline — just swap `--model_name` and `--dataset_name` (with a tweaked `format_example` in workflow.py).

## Notes

- Phi-4-mini is the best candidate to demonstrate QLoRA's real value (3.8B won't fit full fine-tune on a T4)
- For gated models (LLaMA), need `HF_TOKEN` — good to show that flow
- Different architectures may need different LoRA target modules — see README for the lookup snippet
