import torch
from typing import Callable, Any, Literal


def compute_group_normalized_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Computes group-normalized rewards (advantages) for a batch of rollouts.

    This function calculates raw rewards for each response in the rollout batch
    using the provided reward_fn. It then normalizes these rewards within their
    respective groups (defined by group_size).

    Two normalization strategies are supported:
    1.  Standard normalization (if normalize_by_std is True):
        A(i) = (r(i) - mean(group_rewards)) / (std(group_rewards) + advantage_eps)
        (This corresponds to Eq. 28 in the context material).
    2.  Mean-only subtraction (if normalize_by_std is False):
        A(i) = r(i) - mean(group_rewards)
        (This corresponds to Eq. 31 in the context material).

    Args:
        reward_fn: A callable that takes a rollout response (str) and a
            ground truth (str) and returns a dictionary with float values,
            minimally containing a "reward" key.
            Expected signature: Callable[[str, str], dict[str, float]]
        rollout_responses: A list of strings representing the rollouts
            from the policy. The length of this list must be
            rollout_batch_size = n_prompts * group_size.
        repeated_ground_truths: A list of strings representing the
            ground truths for the examples. The length of this list must also
            be rollout_batch_size, as the ground truth for each prompt is
            repeated group_size times to align with the rollout_responses.
        group_size: The number of responses generated for each prompt (i.e.,
            the size of the group for normalization).
        advantage_eps: A small float constant added to the standard
            deviation during normalization (if using std) to prevent
            division by zero.
        normalize_by_std: A boolean flag.
            If True, normalize by subtracting the mean and dividing by the
            standard deviation (plus advantage_eps).
            If False, normalize by only subtracting the group mean.

    Returns:
        A tuple containing:
        - advantages (torch.Tensor): A 1D tensor of shape
            (rollout_batch_size,) containing the group-normalized rewards
            (advantages) for each rollout response.
        - raw_rewards (torch.Tensor): A 1D tensor of shape
            (rollout_batch_size,) containing the unnormalized (raw) rewards
            for each rollout response, as returned by the reward_fn.
        - metadata (dict[str, Any]): A dictionary containing other useful
            statistics for logging, such as the mean/std/min/max of raw
            rewards and advantages.
    """
    rewards = [reward_fn(r, gt)["reward"]
               for r, gt in zip(rollout_responses, repeated_ground_truths)]

    rewards = torch.tensor(rewards)
    rewards_2D_view = rewards.view(-1, group_size)


    advantages = rewards_2D_view - rewards_2D_view.mean(dim=-1, keepdim=True)

    if normalize_by_std:
        advantages = advantages / (rewards_2D_view.std(dim=-1, keepdim=True) + advantage_eps)

    advantages = advantages.flatten()


    metadata = {
            "raw_reward/mean": rewards.mean().item(),
            "raw_reward/std": rewards.std().item(),
            "raw_reward/min": rewards.min().item(),
            "raw_reward/max": rewards.max().item(),
        }

    return advantages, rewards, metadata




def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    """Compute the naive per-token policy-gradient loss.

    This loss is calculated as:
        -A_t * log p_theta(o_t | q, o_<t)
    where A_t is the advantage (or raw reward) and p_theta is the policy.

    Args:
        raw_rewards_or_advantages: torch.Tensor Shape (batch_size, 1), scalar
            reward/advantage for each rollout response. This will be broadcasted
            across the sequence length.
        policy_log_probs: torch.Tensor Shape (batch_size, sequence_length),
            log-probabilities for each token in the sequence.

    Returns:
        torch.Tensor Shape (batch_size, sequence_length), the per-token
        policy-gradient loss. This loss is typically negated before use
        (as per the formula) and then aggregated (e.g., summed or averaged)
        across the batch and sequence dimensions in the training loop.
    """
    if raw_rewards_or_advantages.dim() == 1:
        raw_rewards_or_advantages = raw_rewards_or_advantages.unsqueeze(-1)

    return - raw_rewards_or_advantages * policy_log_probs


def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Computes the per-token GRPO-Clip loss (Equation 33).

    The loss is defined as:
    loss = -min(r_t * A, clip(r_t, 1 - ϵ, 1 + ϵ) * A)
    where:
    - r_t = exp(policy_log_probs - old_log_probs) is the probability ratio.
    - A is the per-example advantage (broadcasted).
    - ϵ is the cliprange.

    This function calculates the loss per token.

    Args:
        advantages: torch.Tensor. Shape (batch_size, 1), representing the
            per-example advantages A. These will be broadcast across the
            sequence length.
        policy_log_probs: torch.Tensor. Shape (batch_size, sequence_length),
            containing the per-token log probabilities from the policy
            being trained (log πθ).
        old_log_probs: torch.Tensor. Shape (batch_size, sequence_length),
            containing the per-token log probabilities from the old
            policy (log πθ_old).
        cliprange: float. The clip parameter ϵ (e.g., 0.2).

    Returns:
        A tuple containing:
        - loss (torch.Tensor): A tensor of shape (batch_size, sequence_length)
          containing the per-token clipped GRPO loss.
        - metadata (dict[str, torch.Tensor]): A dictionary with logging
          information. Contains 'was_clipped', a boolean tensor of shape
          (batch_size, sequence_length) indicating whether the clipped
          term (term2) was smaller than the unclipped term (term1),
          meaning the loss was clipped.
    """

    if advantages.dim() == 1:
        advantages.unsqueeze(-1)

    ratio = torch.exp(policy_log_probs - old_log_probs) # batch_size, seq_len
    a = ratio * advantages
    a_clip = torch.clip(ratio, min=1 - cliprange, max=1 + cliprange) * advantages

    return - torch.min(a, a_clip), {
        "clipped": a > a_clip
    }



def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Select and compute the desired policy-gradient loss.

    This function acts as a convenience wrapper to dispatch to one of three
    policy gradient loss computations based on the `loss_type`:

    (a) "no_baseline": Naive policy gradient (REINFORCE) using raw rewards
        as the advantage (A = R). Delegates to
        `compute_naive_policy_gradient_loss`.
    (b) "reinforce_with_baseline": Naive policy gradient (REINFORCE) using
        pre-computed advantages (A = ¯r, e.g., group-normalized rewards).
        Delegates to `compute_naive_policy_gradient_loss`.
    (c) "grpo_clip": GRPO-Clip loss, which clips the probability ratio.
        Delegates to `compute_grpo_clip_loss`.

    Args:
        policy_log_probs (torch.Tensor): Per-token log-probabilities from the
            policy being trained. Shape: (batch_size, sequence_length).
        loss_type (Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"]):
            The type of policy gradient loss to compute.
        raw_rewards (torch.Tensor | None, optional): Raw rewards. Required if
            `loss_type == "no_baseline"`. Shape: (batch_size, 1).
            Defaults to None.
        advantages (torch.Tensor | None, optional): Pre-computed advantages.
            Required for `loss_type == "reinforce_with_baseline"` and
            `loss_type == "grpo_clip"`. Shape: (batch_size, 1).
            Defaults to None.
        old_log_probs (torch.Tensor | None, optional): Log-probabilities from the
            policy before the update (used for importance sampling). Required
            for `loss_type == "grpo_clip"`.
            Shape: (batch_size, sequence_length). Defaults to None.
        cliprange (float | None, optional): The ϵ value for clipping in
            GRPO-Clip. Required for `loss_type == "grpo_clip"`.
            Defaults to None.

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]]:
            - loss (torch.Tensor): The per-token policy gradient loss.
              Shape: (batch_size, sequence_length).
            - metadata (dict[str, torch.Tensor]): A dictionary containing
              auxiliary statistics from the underlying loss routine (e.g.,
              "clip_frac" for GRPO-Clip).

    Raises:
        AssertionError: If the required arguments for the specified `loss_type`
            are not provided (i.e., are None).
    """
    if loss_type == "no_baseline":
        assert raw_rewards is not None
        return compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs), {}

    if loss_type == "reinforce_with_baseline":
        assert advantages is not None
        return compute_naive_policy_gradient_loss(advantages, policy_log_probs), {}

    if loss_type == "grpo_clip":
        assert advantages is not None
        assert old_log_probs is not None
        assert cliprange is not None
        return compute_grpo_clip_loss(advantages, policy_log_probs, old_log_probs, cliprange)


def masked_mean(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
) -> torch.Tensor:
    """
    Compute the mean of tensor along a given dimension, considering only those
    elements where mask == 1.

    Args:
        tensor (torch.Tensor): The data to be averaged.
        mask (torch.Tensor): Same shape as tensor; positions with 1 are included
            in the mean, 0s are excluded.
        dim (int | None, optional): Dimension over which to average. If None,
            compute the mean over all masked elements, returning a scalar.
            Defaults to None.

    Returns:
        torch.Tensor: The masked mean. The shape matches tensor.mean(dim)
                      semantics (i.e., the specified dim is reduced).

    Raises:
        AssertionError: If tensor and mask do not have the same shape.
    """

    # Implementation tips:
    # 1. Assert tensor.shape == mask.shape
    # 2. Ensure mask is converted to the same dtype as the tensor
    #    (or float) to be used in multiplication.
    # 3. Multiply the tensor by the mask to zero out unselected elements.
    # 4. Sum the result (e.g., torch.sum(tensor * mask, dim=dim))
    # 5. Sum the mask itself along the same dim to get the count of
    #    included elements (e.g., torch.sum(mask, dim=dim))
    # 6. Divide the sum of elements by the count. Handle potential
    #    division by zero (e.g., if a slice has no masked-in elements)
    #    by adding a small epsilon or using torch.where.

    return torch.sum(tensor * mask, dim=dim) / torch.sum(mask, dim=dim)


def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Execute a forward-and-backward pass on a single microbatch for GRPO.

    This function computes the specified policy gradient loss, averages it
    using the response mask, scales the loss for gradient accumulation, and
    performs the backward pass.

    Args:
        policy_log_probs (torch.Tensor): Per-token log-probabilities from the
            policy being trained. Shape: (batch_size, sequence_length).
        response_mask (torch.Tensor): Boolean or {0, 1} tensor. 1 for
            response tokens, 0 for prompt/padding tokens. Used for masking
            the loss. Shape: (batch_size, sequence_length).
        gradient_accumulation_steps (int): The total number of microbatches
            that constitute a single optimizer step.
        loss_type (Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"]):
            The type of policy gradient loss to compute.
        raw_rewards (torch.Tensor | None, optional): Raw rewards. Required if
            `loss_type == "no_baseline"`. Shape: (batch_size, 1).
            Defaults to None.
        advantages (torch.Tensor | None, optional): Pre-computed advantages.
            Required for `loss_type == "reinforce_with_baseline"` and
            `loss_type == "grpo_clip"`. Shape: (batch_size, 1).
            Defaults to None.
        old_log_probs (torch.Tensor | None, optional): Log-probabilities from the
            policy before the update. Required for `loss_type == "grpo_clip"`.
            Shape: (batch_size, sequence_length). Defaults to None.
        cliprange (float | None, optional): The clip parameter ϵ for
            GRPO-Clip. Required for `loss_type == "grpo_clip"`.
            Defaults to None.

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]]:
            - loss (torch.Tensor): The scalar loss for this microbatch,
              scaled by `1.0 / gradient_accumulation_steps`. This value
              is ready for logging.
            - metadata (dict[str, torch.Tensor]): A dictionary containing
              metadata from the underlying loss computation (e.g.,
              "clip_frac") and any other desired statistics.
    """

    # Per the implementation tips, the logic here would:
    # 1. Call compute_policy_gradient_loss(...) to get the per-token loss
    #    and metadata.
    #    loss_per_token, metadata = compute_policy_gradient_loss(...)
    #
    # 2. Compute the masked mean of the loss using the response_mask.
    #    This gives the average loss *per token* across the microbatch.
    #    microbatch_loss = masked_mean(loss_per_token, response_mask)
    #
    # 3. Scale the loss for gradient accumulation.
    #    scaled_loss = microbatch_loss / gradient_accumulation_steps
    #
    # 4. Perform the backward pass.
    #    scaled_loss.backward()
    #
    # 5. Return the detached loss (for logging) and metadata.
    #    return scaled_loss.detach(), metadata
    per_token_loss, metadata = compute_policy_gradient_loss(
        policy_log_probs=policy_log_probs,
        loss_type=loss_type,
        raw_rewards=raw_rewards,
        advantages=advantages,
        old_log_probs=old_log_probs,
        cliprange=cliprange
    ) # batch_size, seq_len

    masked_mean_loss = masked_mean(per_token_loss, response_mask)

    loss = masked_mean_loss / gradient_accumulation_steps
    loss.backward()

    return loss.detach(), metadata