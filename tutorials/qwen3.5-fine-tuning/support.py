"""The task: a support ticket in, a JSON reply out. Everything task-specific lives here.

    {"intent": "<one of 27>", "reply": "<the answer, in the house voice>"}

Swap this module (load, format, score) to fine-tune the same model on something else; the
pipeline in workflow.py does not know what a ticket is.

Scoring is two recomputable numbers per row plus one optional judged one:
    intent       exact match against the gold label (27 classes, the thing the base model guesses at)
    format       the reply is valid JSON with both keys, and keeps every {{placeholder}} the ticket used
    judge        optional: a pairwise, position-swapped LLM judge for the reply's voice (see workflow.py)
"""

from __future__ import annotations

import json
import re

INTENTS = [
    "cancel_order", "change_order", "change_shipping_address", "check_cancellation_fee", "check_invoice",
    "check_payment_methods", "check_refund_policy", "complaint", "contact_customer_service", "contact_human_agent",
    "create_account", "delete_account", "delivery_options", "delivery_period", "edit_account", "get_invoice",
    "get_refund", "newsletter_subscription", "payment_issue", "place_order", "recover_password",
    "registration_problems", "review", "set_up_shipping_address", "switch_account", "track_order", "track_refund",
]

SYSTEM_PROMPT = (
    "You are the support assistant for an online store. Classify the customer's message into exactly one intent "
    "and write the reply.\nIntents: " + ", ".join(INTENTS) + "\n"
    'Reply with JSON only: {"intent": "<intent>", "reply": "<reply>"}. Keep any {{placeholders}} from the message as they are.'
)

PLACEHOLDER = re.compile(r"\{\{[^}]+\}\}")


def messages(ticket: str) -> list[dict]:
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": ticket}]


def target(example: dict) -> str:
    """The completion the model should produce for a training row."""
    return json.dumps({"intent": example["intent"], "reply": example["response"]}, ensure_ascii=False)


def render(tokenizer, msgs: list[dict], add_generation_prompt: bool) -> str:
    """Qwen3.5's template thinks by default; the answer is a JSON object, so thinking is off."""
    kwargs = dict(tokenize=False, add_generation_prompt=add_generation_prompt)
    try:
        return tokenizer.apply_chat_template(msgs, enable_thinking=False, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(msgs, **kwargs)


def training_pair(tokenizer, example: dict) -> tuple[str, str]:
    prompt = render(tokenizer, messages(example["instruction"]), add_generation_prompt=True)
    return prompt, target(example) + tokenizer.eos_token


def generation_prompt(tokenizer, example: dict) -> str:
    return render(tokenizer, messages(example["instruction"]), add_generation_prompt=True)


# ------------------------------------------------------------------
# Scoring
# ------------------------------------------------------------------


def parse(output: str) -> dict | None:
    """The first JSON object in the output, or None. Code fences and a stray <think> block are tolerated."""
    text = re.sub(r"<think>.*?</think>", "", output, flags=re.DOTALL)
    text = text.replace("```json", "").replace("```", "")
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def score(output: str, example: dict) -> dict:
    obj = parse(output)
    intent = (obj or {}).get("intent", "")
    reply = (obj or {}).get("reply", "")
    valid = obj is not None and isinstance(reply, str) and bool(reply.strip()) and intent in INTENTS
    needed = set(PLACEHOLDER.findall(example["instruction"]))
    kept = needed.issubset(set(PLACEHOLDER.findall(reply))) if isinstance(reply, str) else False
    return {
        "output": output,
        "intent": intent,
        "reply": reply if isinstance(reply, str) else "",
        "intent_ok": intent == example["intent"],
        "format_ok": valid and kept,
        "json_ok": obj is not None,
    }


def summarize(rows: list[dict]) -> dict:
    n = max(len(rows), 1)
    return {
        "intent_accuracy": round(sum(r["intent_ok"] for r in rows) / n * 100, 1),
        "format_rate": round(sum(r["format_ok"] for r in rows) / n * 100, 1),
        "json_rate": round(sum(r["json_ok"] for r in rows) / n * 100, 1),
    }


# ------------------------------------------------------------------
# LoRA targets, discovered rather than guessed
# ------------------------------------------------------------------

# Qwen3.5 is a hybrid: three Gated DeltaNet blocks for every attention block, plus a vision encoder
# this tutorial never uses. Adapting "q_proj, k_proj, v_proj, o_proj" would skip three quarters of
# the model; adapting the vision tower would waste parameters on frozen pixels.
_SKIP_SUFFIXES = ("lm_head", "embed_tokens", "router", "shared_expert_gate")
_SKIP_PATHS = ("visual", "vision")


def discover_lora_targets(model) -> list[str]:
    import torch.nn as nn

    names: set[str] = set()
    for full_name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if any(p in full_name for p in _SKIP_PATHS):
            continue
        leaf = full_name.rsplit(".", 1)[-1]
        if any(leaf.endswith(s) for s in _SKIP_SUFFIXES):
            continue
        names.add(leaf)
    return sorted(names)
