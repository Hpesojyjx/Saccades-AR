"""Small, device-agnostic scheduling primitives shared by VAR train/inference."""

import torch


def random_unfixed(fixed_mask, count, generator=None):
    batch = fixed_mask.shape[0]
    selected = torch.empty(batch, count, dtype=torch.long, device=fixed_mask.device)
    for batch_index in range(batch):
        available = (~fixed_mask[batch_index]).nonzero(as_tuple=True)[0]
        if count > len(available):
            raise ValueError("cannot select more positions than remain unfixed")
        permutation = torch.randperm(
            len(available), device=fixed_mask.device, generator=generator
        )[:count]
        selected[batch_index] = available[permutation]
    return selected


def nearest_to_anchor(hidden, fixed_mask, anchor, count):
    distances = ((hidden - anchor.unsqueeze(1)) ** 2).sum(dim=-1)
    distances = distances.masked_fill(fixed_mask, float("inf"))
    return distances.topk(count, dim=1, largest=False).indices
