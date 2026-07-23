"""The GRPO objective, unpacked.

At Level 1 TRL's GRPOTrainer owns this and you never see it. At Level 2 the loop
is ours, so the math has to be explicit. It is not much math — that is rather the
point of GRPO. There is no value network and no GAE; the baseline is just the mean
reward of the other completions for the same prompt.

Everything here is a pure function so it can be unit-tested without a GPU.
`group_advantages` in particular is worth reading before running anything: it is
where "my reward is going up but nothing is learning" gets diagnosed.
"""

import logging
from dataclasses import dataclass

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


@dataclass
class AdvantageStats:
    """Diagnostics for one round's advantage computation.

    `dead_groups` is the number to watch. A group where every completion got the
    same reward — all passed, or (far more often) all failed — has zero variance,
    therefore zero advantage, therefore zero gradient. It costs a full generation
    and verification cycle and teaches the model nothing.

    If dead_groups is close to total_groups, no hyperparameter will save the run.
    Either the base model is too weak to ever land in the middle band, or the
    problems are too hard. Change the model or the data, not the learning rate.
    """
    total_groups: int
    dead_groups: int
    mean_reward: float
    solved_groups: int   # groups where at least one completion passed

    @property
    def live_fraction(self) -> float:
        return (self.total_groups - self.dead_groups) / max(1, self.total_groups)


def group_advantages(
    rewards: list[float],
    group_size: int,
    eps: float = 1e-4,
) -> tuple[list[float], AdvantageStats]:
    """Group-relative advantages: A_i = (r_i - mean(group)) / (std(group) + eps).

    `rewards` must be laid out as consecutive groups of `group_size`, i.e. all
    completions for prompt 0, then all for prompt 1, and so on.

    Normalizing *within* the group is the whole trick. The baseline is the group
    mean, so a completion is reinforced for being better than its siblings on the
    same problem — not better than some global average across problems of wildly
    different difficulty. That is what removes the need for a value network.

    A zero-variance group yields all-zero advantages (not NaN — note the eps in the
    denominator applies to a zero numerator, so the result is exactly 0.0).
    """
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    if len(rewards) % group_size != 0:
        raise ValueError(
            f"len(rewards)={len(rewards)} is not a multiple of group_size={group_size}; "
            "rewards must be laid out as consecutive complete groups"
        )

    advantages: list[float] = []
    dead = 0
    solved = 0
    n_groups = len(rewards) // group_size

    for g in range(n_groups):
        group = rewards[g * group_size:(g + 1) * group_size]
        mean = sum(group) / group_size
        var = sum((r - mean) ** 2 for r in group) / group_size
        std = var ** 0.5

        if any(r > 0 for r in group):
            solved += 1

        if std < 1e-8:
            # Flat group — no signal. Emit zeros rather than dividing by eps, which
            # would amplify float noise into spurious gradients.
            dead += 1
            advantages.extend([0.0] * group_size)
            continue

        advantages.extend([(r - mean) / (std + eps) for r in group])

    stats = AdvantageStats(
        total_groups=n_groups,
        dead_groups=dead,
        mean_reward=sum(rewards) / max(1, len(rewards)),
        solved_groups=solved,
    )
    return advantages, stats


def sequence_logprobs(model, input_ids, attention_mask, completion_mask):
    """Sum of per-token log-probs over the *completion* tokens of each sequence.

    Args:
        input_ids: (B, T) prompt + completion, right-padded.
        attention_mask: (B, T) 1 for real tokens.
        completion_mask: (B, T) 1 for tokens that are part of the completion —
            prompt tokens are excluded, because the policy did not choose them and
            reinforcing them would just teach the model to memorize the prompts.

    Returns (per_sequence_sum, per_token_logprobs, shifted_mask). The shifted mask
    comes back because the targets are offset by one, so the caller must use the
    shifted version rather than the mask it passed in.
    """
    import torch

    out = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = out.logits[:, :-1, :]          # predict token t+1 from position t
    targets = input_ids[:, 1:]
    mask = completion_mask[:, 1:].to(logits.dtype)

    logprobs = torch.log_softmax(logits.float(), dim=-1)
    token_lp = torch.gather(logprobs, 2, targets.unsqueeze(-1)).squeeze(-1)
    token_lp = token_lp * mask
    return token_lp.sum(dim=1), token_lp, mask


def grpo_loss(
    token_lp_new,
    token_lp_old,
    token_lp_ref,
    mask,
    advantages,
    beta: float = 0.04,
    clip_eps: float = 0.2,
):
    """Clipped surrogate objective + KL anchor to the reference (base) policy.

        loss = -mean_i[ 1/|o_i| * sum_t min(r_t * A_i, clip(r_t, 1-e, 1+e) * A_i) ]
               + beta * KL(policy || ref)

    where `r_t = exp(logp_new_t - logp_old_t)` is a per-token ratio and `A_i` is the
    group-relative advantage of sequence i, shared by all of its tokens.

    The KL term uses the k3 estimator, `exp(d) - d - 1` where `d = logp_ref - logp`.
    It is unbiased and always non-negative, unlike the naive `-d` which can go
    negative on a minibatch and push the policy *away* from the reference.

    Why anchor at all: with a weak or absent KL leash the policy drifts wherever
    the reward points, including into degenerate behavior that happens to pass
    tests. The signature symptom is train reward climbing while held-out pass rate
    drops. beta=0.04 is a reasonable default; beta=0 disables it.

    NOTE ON READING THE LOSS: with `inner_epochs=1` the returned policy loss is
    ~0.0 at *every* step, and that is correct, not a bug. Advantages are zero-mean
    within each group and the on-policy ratio is exactly 1, so the loss *value* is
    `-mean(A) = 0` while its *gradient* is not. Do not tune against this number —
    it carries no signal. Watch mean reward and the live-group count instead.
    """
    import torch

    # Per-TOKEN importance ratio, with the sequence's advantage broadcast across
    # its tokens. This is the DeepSeekMath formulation, and the choice is not
    # cosmetic: a sequence-level ratio is exp(SUM of per-token log-prob diffs), so
    # on a 192-token completion a drift of 0.05 nats per token gives exp(9.6) ~
    # 15000. The clip band is blown past instantly and the gradient is garbage.
    # Token-level ratios stay near 1 regardless of completion length.
    ratio = torch.exp(torch.clamp(token_lp_new - token_lp_old, min=-20.0, max=20.0))

    adv = advantages.to(ratio.dtype).unsqueeze(1)  # (B, 1) -> broadcast over tokens
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
    per_token_loss = -torch.min(unclipped, clipped) * mask

    # Normalize by each sequence's own length before averaging over sequences (the
    # 1/|o_i| in the GRPO objective). Summing over tokens instead would weight long
    # completions more heavily and hand the model a length bias it can exploit.
    seq_lens = mask.sum(dim=1).clamp(min=1.0)
    policy_loss = (per_token_loss.sum(dim=1) / seq_lens).mean()

    kl = torch.zeros((), device=policy_loss.device, dtype=policy_loss.dtype)
    if beta > 0:
        d = (token_lp_ref - token_lp_new).clamp(min=-20.0, max=20.0)
        k3 = (torch.exp(d) - d - 1.0) * mask
        n_tokens = mask.sum().clamp(min=1.0)
        kl = k3.sum() / n_tokens

    # Mean ratio over real tokens only — a scalar diagnostic. It should sit at
    # exactly 1.0 on the first inner epoch and drift away as the batch goes stale.
    mean_ratio = (ratio * mask).sum() / mask.sum().clamp(min=1.0)

    return policy_loss + beta * kl, policy_loss.detach(), kl.detach(), mean_ratio.detach()
