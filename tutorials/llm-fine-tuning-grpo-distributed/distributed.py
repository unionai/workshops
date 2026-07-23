"""
Level 2 — Fully disaggregated GRPO.

Level 1 fanned out verification but left generation on the training GPU. Here the
loop itself is Flyte's, and each phase runs where it belongs:

    per round:
      1. learner  : save the current LoRA adapter        -> flyte.io.Dir
      2. rollouts : fan out to the vLLM pool             (rollout.py, L40s, reusable)
      3. verify   : fan out to the sandbox pool          (verify.py, CPU, reusable)
      4. roll up  : (prompt, completion, logprob, reward)
      5. learner  : group advantages -> GRPO loss -> one optimizer step

The learner is a single long-lived task that drives all of this, rather than one
task per round. That is deliberate: it loads a 14B base once and keeps optimizer
state in memory across rounds. Splitting it into per-round tasks would mean
reloading the model every round and either losing Adam's moments or serializing
them to blob storage each time.

Weight sync is the part no orchestrator does for you. Here it is explicit and
about six lines: the learner writes the adapter, the rollout workers load it by
path. That is cheap only because we train LoRA — the adapter is tens of MB while
the base is ~28GB, and the base never moves.

Usage:
    flyte run distributed.py distributed_pipeline
    flyte run distributed.py distributed_pipeline --rounds 3 --prompts_per_round 16 \\
        --rollout_workers 2
"""

import asyncio
import json
import logging
import os
import random
import tempfile
import time

import flyte
import flyte.io
import flyte.report

from common import (
    build_code_prompt,
    download_model,
    func_def_from_prompt,
    prepare_data,
)
from config import cpu_env, learner_env
from learner import grpo_loss, group_advantages, sequence_logprobs
from report_helpers import make_line_chart, pipeline_step_indicator, wrap_report
from rollout import RolloutRequest, RolloutSample, generate_rollouts
from verify import VerifyItem, chunk, verify_shard
from workflow import evaluate

logging.basicConfig(level=logging.WARNING, format="%(message)s", force=True)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def _disable_dropout(model) -> int:
    """Zero every dropout probability in the model. Returns how many it changed.

    This is not a performance tweak, it is a correctness requirement for the
    importance ratio. GRPO weights each token by
    `exp(logp_new - logp_old)`, which is only meaningful if the two forward passes
    differ *because the parameters changed*. With dropout active they also differ
    by a fresh random mask, so the ratio wobbles around 1 on a batch that has had
    no update at all, and that noise goes straight into the gradient.

    RLHF/RLVR implementations disable dropout during the policy update for exactly
    this reason. With it off, the first inner epoch has ratio == 1.0 exactly, and
    any departure from 1.0 is real policy movement rather than sampling noise.
    """
    import torch

    changed = 0
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout) and module.p != 0.0:
            module.p = 0.0
            changed += 1
    return changed


def _build_batch(tokenizer, pairs, device, pad_id):
    """Tokenize (prompt, completion) pairs into a right-padded batch.

    Returns input_ids, attention_mask, completion_mask. The completion mask marks
    only the generated tokens: the policy did not choose the prompt tokens, and
    reinforcing them would teach the model to memorize prompts rather than to
    solve them.
    """
    import torch

    seqs = []
    for prompt, completion in pairs:
        p_ids = tokenizer(prompt, add_special_tokens=True).input_ids
        c_ids = tokenizer(completion, add_special_tokens=False).input_ids
        if not c_ids:
            c_ids = [pad_id]  # empty generation — keep the row, it scores 0 anyway
        seqs.append((p_ids, c_ids))

    max_len = max(len(p) + len(c) for p, c in seqs)
    input_ids, attn, comp_mask = [], [], []
    for p_ids, c_ids in seqs:
        ids = p_ids + c_ids
        pad = max_len - len(ids)
        input_ids.append(ids + [pad_id] * pad)
        attn.append([1] * len(ids) + [0] * pad)
        comp_mask.append([0] * len(p_ids) + [1] * len(c_ids) + [0] * pad)

    return (
        torch.tensor(input_ids, device=device),
        torch.tensor(attn, device=device),
        torch.tensor(comp_mask, device=device),
    )


@learner_env.task(report=True)
async def train_distributed(
    model_name: str,
    model_dir: flyte.io.Dir,
    data_dir: flyte.io.Dir,
    rounds: int = 8,
    prompts_per_round: int = 16,
    num_generations: int = 8,
    rollout_workers: int = 2,
    lr: float = 5e-5,
    beta: float = 0.04,
    clip_eps: float = 0.2,
    inner_epochs: int = 1,
    lora_r: int = 16,
    lora_alpha: int = 32,
    max_completion_length: int = 192,
    micro_batch_size: int = 8,
    shard_size: int = 4,
    use_chat_template: bool = False,
    temperature: float = 0.9,
) -> flyte.io.Dir:
    """Run the disaggregated GRPO loop and return the trained LoRA adapter.

    Args:
        rounds: policy updates. Each round = one full rollout + verify + step cycle.
        prompts_per_round: problems sampled per round. Total completions per round
            is `prompts_per_round * num_generations`.
        rollout_workers: how many parallel `generate_rollouts` calls to split the
            prompts across. Capped in practice by rollout_env's max replicas.
        inner_epochs: gradient steps per rollout batch. 1 keeps the update strictly
            on-policy (the importance ratio is exactly 1 and clipping never
            engages). >1 reuses the batch and is where the clipped surrogate starts
            doing real work — at the cost of the data going stale.
        micro_batch_size: sequences per forward pass. The whole round's completions
            will not fit on one card at 14B; this is the knob that keeps it inside
            memory.
    """
    import torch
    from datasets import load_from_disk
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log.info(f"GRPO (Level 2, disaggregated): {model_name}, {rounds} rounds")

    await flyte.report.replace.aio(
        wrap_report(f"<h2>Loading Learner...</h2><h3>{model_name}</h3>"), do_flush=True
    )

    data_path = await data_dir.download()
    dataset = load_from_disk(data_path)
    train_ds = dataset["train"]

    model_path = await model_dir.download()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_id = tokenizer.pad_token_id

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=dtype, device_map="auto", attn_implementation="eager"
    )
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        # 0.0, not the 0.05 you would use for supervised fine-tuning — see
        # _disable_dropout above. Dropout breaks the importance ratio.
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    n_dropout = _disable_dropout(model)
    log.info(f"Disabled {n_dropout} dropout layers (required for a meaningful importance ratio)")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr
    )

    device = next(model.parameters()).device
    history: list[dict] = []
    rng = random.Random(0)

    def _report(current_round: int) -> str:
        last = history[-1] if history else {}
        html_out = f"""
        <h2>Disaggregated GRPO — Round {current_round}/{rounds}</h2>
        <h3>{model_name}</h3>
        <div class="stat-grid">
          <div class="stat"><div class="value">{last.get('mean_reward', 0):.3f}</div><div class="label">Mean Reward</div></div>
          <div class="stat"><div class="value">{last.get('live_groups', 0)}/{last.get('total_groups', 0)}</div><div class="label">Live Groups</div></div>
          <div class="stat"><div class="value">{rollout_workers}</div><div class="label">Rollout Workers</div></div>
          <div class="stat"><div class="value">{last.get('rollout_s', 0):.0f}s</div><div class="label">Rollout</div></div>
          <div class="stat"><div class="value">{last.get('verify_s', 0):.0f}s</div><div class="label">Verify</div></div>
          <div class="stat"><div class="value">{last.get('update_s', 0):.0f}s</div><div class="label">Update</div></div>
        </div>
        """
        if len(history) > 1:
            html_out += (
                '<div class="chart-container">'
                + make_line_chart(
                    history, x_key="round",
                    y_keys=["mean_reward", "live_fraction"],
                    title="Reward and live-group fraction by round",
                    colors=["#0f3460", "#adb5bd"],
                )
                + "</div>"
            )
        # Live groups is the number to watch, not reward. A round where every group
        # is flat produced no gradient at all, however good the reward looks.
        html_out += (
            '<div class="card"><p><b>Live groups</b> are groups with reward variance — '
            "the only ones that produce a gradient. If this collapses toward zero, the "
            "base model is not landing in the middle band and no hyperparameter will fix it.</p>"
            "<p>The <b>policy loss is ~0 at every step</b> with <code>inner_epochs=1</code>, and "
            "that is expected: advantages are zero-mean within a group and the on-policy ratio "
            "is exactly 1, so the loss <i>value</i> vanishes while its <i>gradient</i> does not. "
            "Read mean reward and live groups, not the loss.</p></div>"
        )
        return wrap_report(html_out)

    for rnd in range(rounds):
        round_started = time.monotonic()

        # -- 1. Publish the current policy ------------------------------------
        # LoRA initializes B=0, so at round 0 the adapter is an exact identity and
        # the rollout workers generate from the untouched base model. That gives a
        # clean round-0 baseline without a separate code path.
        adapter_local = os.path.join(tempfile.mkdtemp(), f"adapter-r{rnd}")
        model.save_pretrained(adapter_local)
        adapter_dir = await flyte.io.Dir.from_local(adapter_local)

        # -- 2. Fan out rollouts ----------------------------------------------
        sample_idx = rng.sample(range(len(train_ds)), min(prompts_per_round, len(train_ds)))
        problems = [train_ds[i] for i in sample_idx]
        requests = [
            RolloutRequest(
                index=i,
                prompt=build_code_prompt(tokenizer, p["func_prompt"], use_chat_template),
            )
            for i, p in enumerate(problems)
        ]

        worker_chunks = chunk(requests, max(1, -(-len(requests) // max(1, rollout_workers))))
        t0 = time.monotonic()
        rollout_results = await asyncio.gather(*(
            generate_rollouts(
                wc, model_dir, adapter_dir, rnd, num_generations,
                max_completion_length, temperature, 0.95, lora_r,
            )
            for wc in worker_chunks
        ))
        rollout_s = time.monotonic() - t0
        samples = [s for group in rollout_results for s in group]

        # Regroup by prompt index. The fan-out returns shards in submission order,
        # but advantages are computed *within* a prompt's group — so the layout
        # here has to be exactly `group_size` consecutive completions per prompt.
        by_index: dict[int, list] = {i: [] for i in range(len(requests))}
        for s in samples:
            by_index[s.index].append(s)

        ordered, pairs, verify_items = [], [], []
        for i, prob in enumerate(problems):
            group = by_index[i][:num_generations]
            # A worker can return short if a generation was truncated or dropped.
            # Pad with empty completions so every group is exactly num_generations —
            # a ragged layout would silently misalign every later group.
            while len(group) < num_generations:
                group.append(RolloutSample(index=i, completion="", logprob=0.0, num_tokens=0))
            for s in group:
                ordered.append(s)
                pairs.append((requests[i].prompt, s.completion))
                verify_items.append(VerifyItem(
                    func_def=func_def_from_prompt(prob["func_prompt"]),
                    completion=s.completion,
                    setup=prob["setup_code"] or "",
                    tests=prob["tests"],
                ))

        # -- 3. Fan out verification ------------------------------------------
        shards = chunk(verify_items, shard_size)
        t0 = time.monotonic()
        shard_results = await asyncio.gather(*(verify_shard(s) for s in shards))
        verify_s = time.monotonic() - t0
        rewards = [r.reward for shard in shard_results for r in shard]

        # -- 4. Roll up: advantages -------------------------------------------
        advantages, adv_stats = group_advantages(rewards, num_generations)
        log.info(
            f"[round {rnd}] reward={adv_stats.mean_reward:.3f} "
            f"live={adv_stats.total_groups - adv_stats.dead_groups}/{adv_stats.total_groups} "
            f"rollout={rollout_s:.0f}s verify={verify_s:.0f}s"
        )

        if adv_stats.dead_groups == adv_stats.total_groups:
            # Every group flat — the gradient would be exactly zero. Skip the
            # update rather than burn a forward/backward pass on nothing.
            log.warning(f"[round {rnd}] all {adv_stats.total_groups} groups flat; skipping update")
            history.append({
                "round": rnd, "mean_reward": adv_stats.mean_reward,
                "live_fraction": 0.0, "live_groups": 0,
                "total_groups": adv_stats.total_groups, "loss": 0.0,
                "rollout_s": rollout_s, "verify_s": verify_s, "update_s": 0.0,
            })
            await flyte.report.replace.aio(_report(rnd + 1), do_flush=True)
            continue

        # -- 5. Policy update --------------------------------------------------
        t0 = time.monotonic()
        adv_t = torch.tensor(advantages, device=device, dtype=torch.float32)

        micro = [
            (i, min(i + micro_batch_size, len(pairs)))
            for i in range(0, len(pairs), micro_batch_size)
        ]

        # Cache the pre-update log-probs (policy and reference) once per round.
        # `logp_old` comes from our own forward pass, not from vLLM's: the two
        # implementations differ numerically, and using vLLM's numbers as the PPO
        # denominator would inject that mismatch straight into the gradient. The
        # vLLM log-probs are used below only as a drift diagnostic.
        # No model.eval()/model.train() dance here: dropout was zeroed at load time
        # (_disable_dropout), so the no-grad caching pass and the update pass are
        # numerically identical forwards. Staying in train mode also keeps gradient
        # checkpointing on its normal path.
        cached = []
        with torch.no_grad():
            for lo, hi in micro:
                ids, attn, cmask = _build_batch(tokenizer, pairs[lo:hi], device, pad_id)
                _, tok_old, mask = sequence_logprobs(model, ids, attn, cmask)
                with model.disable_adapter():
                    # Reference = the base model. Disabling the adapter gives it for
                    # free; no second copy of 28GB of weights in memory.
                    _, tok_ref, _ = sequence_logprobs(model, ids, attn, cmask)
                cached.append((ids, attn, cmask, tok_old.detach(), tok_ref.detach(), mask))

        total_loss = 0.0
        total_kl = 0.0
        mean_ratio = 1.0
        for _ in range(inner_epochs):
            optimizer.zero_grad(set_to_none=True)
            ratio_acc = 0.0
            for (lo, hi), (ids, attn, cmask, tok_old, tok_ref, mask) in zip(micro, cached):
                _, tok_new, m = sequence_logprobs(model, ids, attn, cmask)
                loss, pol, kl, ratio = grpo_loss(
                    tok_new, tok_old, tok_ref, m,
                    adv_t[lo:hi], beta=beta, clip_eps=clip_eps,
                )
                # Scale so accumulated micro-batch gradients average, not sum.
                w = (hi - lo) / len(pairs)
                (loss * w).backward()
                total_loss += float(pol) * w
                total_kl += float(kl) * w
                ratio_acc += float(ratio) * w
            mean_ratio = ratio_acc

            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()

        update_s = time.monotonic() - t0

        # Drift diagnostic: how far the rollout engine's log-probs sit from the
        # learner's for the same tokens. Small and stable is healthy. A large or
        # growing gap means generation and training disagree about the policy —
        # usually a dtype or adapter-sync problem, and the first thing to check if
        # reward stops tracking the loss.
        vllm_lp = sum(s.logprob for s in ordered) / max(1, len(ordered))

        history.append({
            "round": rnd,
            "mean_reward": round(adv_stats.mean_reward, 4),
            "live_fraction": round(adv_stats.live_fraction, 4),
            "live_groups": adv_stats.total_groups - adv_stats.dead_groups,
            "total_groups": adv_stats.total_groups,
            "solved_groups": adv_stats.solved_groups,
            "loss": round(total_loss, 5),
            "kl": round(total_kl, 5),
            # Exactly 1.0 with inner_epochs=1. With inner_epochs>1 it drifts as the
            # batch goes stale — that drift is what the clip band exists to bound,
            # and a mean ratio far outside [1-eps, 1+eps] means the batch is being
            # reused past its usefulness.
            "mean_ratio": round(mean_ratio, 5),
            "vllm_mean_logprob": round(vllm_lp, 3),
            "rollout_s": round(rollout_s, 1),
            "verify_s": round(verify_s, 1),
            "update_s": round(update_s, 1),
            "round_s": round(time.monotonic() - round_started, 1),
        })
        await flyte.report.replace.aio(_report(rnd + 1), do_flush=True)

    save_dir = os.path.join(tempfile.mkdtemp(), "final")
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    log.info(f"Training complete over {rounds} rounds.")

    await flyte.report.replace.aio(_report(rounds), do_flush=True)
    return await flyte.io.Dir.from_local(save_dir)


@cpu_env.task(report=True)
async def distributed_pipeline(
    # Not named `pipeline`: Flyte derives a task's registered name from the function
    # name and its environment (`grpo-dist-cpu.pipeline`), so a second `pipeline` in
    # the same environment would collide with workflow.py's and silently overwrite
    # it at registration.
    model_name: str = "Qwen/Qwen2.5-Coder-14B",
    rounds: int = 8,
    prompts_per_round: int = 16,
    num_generations: int = 8,
    rollout_workers: int = 2,
    lr: float = 5e-5,
    beta: float = 0.04,
    clip_eps: float = 0.2,
    inner_epochs: int = 1,
    lora_r: int = 16,
    lora_alpha: int = 32,
    max_completion_length: int = 192,
    micro_batch_size: int = 8,
    shard_size: int = 4,
    max_candidate_samples: int = 300,
    max_eval_samples: int = 50,
    num_eval_examples: int = 50,
    eval_k: int = 4,
    use_chat_template: bool = False,
) -> str:
    """Level 2 pipeline: prepare -> disaggregated GRPO -> evaluate."""
    steps = ["Prepare Data", "Disaggregated GRPO", "Evaluate"]
    log.info(f"Level 2 pipeline: {model_name}, {rounds} rounds")

    await flyte.report.replace.aio(
        wrap_report(
            f"<h2>Disaggregated GRPO — Level 2</h2><h3>{model_name}</h3>"
            f"{pipeline_step_indicator(0, steps)}"
            f'<div class="card"><p>Preparing coding problems...</p></div>'
        ),
        do_flush=True,
    )

    data_dir = await prepare_data(max_candidate_samples, max_eval_samples)
    model_dir = await download_model(model_name)

    await flyte.report.replace.aio(
        wrap_report(
            f"<h2>Disaggregated GRPO — Level 2</h2><h3>{model_name}</h3>"
            f"{pipeline_step_indicator(1, steps)}"
            f'<div class="card"><p>Rollout &rarr; verify &rarr; update, {rounds} rounds...</p></div>'
        ),
        do_flush=True,
    )

    adapter_dir = await train_distributed(
        model_name, model_dir, data_dir, rounds, prompts_per_round, num_generations,
        rollout_workers, lr, beta, clip_eps, inner_epochs, lora_r, lora_alpha,
        max_completion_length, micro_batch_size, shard_size, use_chat_template,
    )

    await flyte.report.replace.aio(
        wrap_report(
            f"<h2>Disaggregated GRPO — Level 2</h2><h3>{model_name}</h3>"
            f"{pipeline_step_indicator(2, steps)}"
            f'<div class="card"><p>Evaluating base vs GRPO...</p></div>'
        ),
        do_flush=True,
    )

    result = await evaluate(
        model_name, model_dir, adapter_dir, data_dir, num_eval_examples,
        use_chat_template, eval_k,
    )
    metrics = json.loads(result)
    improvement = metrics["improvement"]
    badge = "badge-success" if improvement > 0 else "badge-danger" if improvement < 0 else "badge-info"

    await flyte.report.replace.aio(
        wrap_report(
            f"<h2>Disaggregated GRPO — Complete</h2><h3>{model_name}</h3>"
            f"{pipeline_step_indicator(3, steps)}"
            f'<div class="stat-grid">'
            f'  <div class="stat"><div class="value">{metrics["base_pass_rate"]}%</div><div class="label">Base pass@1</div></div>'
            f'  <div class="stat"><div class="value">{metrics["grpo_pass_rate"]}%</div><div class="label">GRPO pass@1</div></div>'
            f'  <div class="stat"><div class="value"><span class="badge {badge}">{improvement:+.1f}pp</span></div><div class="label">Improvement</div></div>'
            f"</div>"
        ),
        do_flush=True,
    )

    return result


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(distributed_pipeline)
    print(run.url)
