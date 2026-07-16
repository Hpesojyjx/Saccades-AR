import torch


def _random(x_B, fixed_mask, tokens_per_iter, rng, **_):
    B, L, _ = x_B.shape
    top_indices = torch.zeros(B, tokens_per_iter, dtype=torch.long, device=x_B.device)
    for b_idx in range(B):
        unfixed_pos = (~fixed_mask[b_idx]).nonzero(as_tuple=True)[0]
        perm = torch.randperm(len(unfixed_pos), device=x_B.device, generator=rng)[:tokens_per_iter]
        top_indices[b_idx] = unfixed_pos[perm]
    return top_indices


def _sar(x_B, fixed_mask, cond_BD, tokens_per_iter, **_):
    """Select positions nearest to Infinity's attentive-pooled text anchor."""
    anchor = cond_BD.unsqueeze(1)
    scores = ((x_B - anchor) ** 2).sum(dim=-1)
    scores = scores.masked_fill(fixed_mask, float('inf'))
    _, top_indices = scores.topk(tokens_per_iter, dim=1, largest=False)
    return top_indices


def _sequential(x_B, fixed_mask, tokens_per_iter, **_):
    """Select the lowest-index positions that have not been committed."""
    batch = x_B.shape[0]
    selected = torch.zeros(
        batch, tokens_per_iter, dtype=torch.long, device=x_B.device
    )
    for batch_index in range(batch):
        available = (~fixed_mask[batch_index]).nonzero(as_tuple=True)[0]
        selected[batch_index] = available[:tokens_per_iter]
    return selected


def _inside_out(x_B, fixed_mask, tokens_per_iter, **_):
    """Select uncommitted positions nearest to the 1D token midpoint."""
    batch, length, _ = x_B.shape
    center = length // 2
    positions = torch.arange(length, device=x_B.device)
    distances = (positions - center).abs().float()
    distances = distances.unsqueeze(0).expand(batch, -1)
    distances = distances.masked_fill(fixed_mask, float("inf"))
    return distances.topk(tokens_per_iter, dim=1, largest=False).indices


# Registry: name -> fn(x_B, fixed_mask, cond_BD, tokens_per_iter, rng, **kwargs) -> top_indices
SCHEDULING_STRATEGIES = {
    'random': _random,
    'sar': _sar,
    'sequential': _sequential,
    'inside_out': _inside_out,
}
