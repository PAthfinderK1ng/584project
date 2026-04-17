from __future__ import annotations

from typing import Iterable

import torch


def _normalize_weights(weights: Iterable[float], count: int) -> torch.Tensor:
    values = list(weights)
    if not values:
        values = [1.0 / count] * count
    tensor = torch.tensor(values, dtype=torch.float32)
    return tensor / tensor.sum()


def linear_merge(tensors: list[torch.Tensor], weights: list[float] | None = None) -> torch.Tensor:
    normalized = _normalize_weights(weights or [], len(tensors))
    output = torch.zeros_like(tensors[0], dtype=torch.float32)
    for weight, tensor in zip(normalized, tensors):
        output = output + weight * tensor.float()
    return output.to(dtype=tensors[0].dtype)


def _trim_tensor(tensor: torch.Tensor, density: float) -> torch.Tensor:
    if density >= 1.0:
        return tensor
    flat = tensor.flatten()
    keep = max(1, int(flat.numel() * density))
    topk = torch.topk(flat.abs(), keep).indices
    mask = torch.zeros_like(flat, dtype=torch.bool)
    mask[topk] = True
    trimmed = torch.where(mask, flat, torch.zeros_like(flat))
    return trimmed.view_as(tensor)


def ties_merge(tensors: list[torch.Tensor], density: float = 0.5) -> torch.Tensor:
    trimmed = [_trim_tensor(tensor.float(), density) for tensor in tensors]
    stacked = torch.stack(trimmed, dim=0)
    majority_sign = torch.sign(stacked.sum(dim=0))
    agreement_mask = torch.sign(stacked) == majority_sign.unsqueeze(0)
    agreement_mask = agreement_mask & (majority_sign.unsqueeze(0) != 0)

    agreed_sum = (stacked * agreement_mask).sum(dim=0)
    agreed_count = agreement_mask.sum(dim=0).clamp_min(1)
    fallback = stacked.mean(dim=0)
    merged = torch.where(agreement_mask.any(dim=0), agreed_sum / agreed_count, fallback)
    return merged.to(dtype=tensors[0].dtype)


def dare_linear_merge(
    tensors: list[torch.Tensor],
    drop_rate: float = 0.5,
    weights: list[float] | None = None,
    seed: int = 42,
) -> torch.Tensor:
    if not 0.0 <= drop_rate < 1.0:
        raise ValueError("drop_rate must be in [0, 1).")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    retained_probability = 1.0 - drop_rate
    stochastically_pruned = []
    for tensor in tensors:
        random_mask = torch.bernoulli(
            torch.full_like(tensor.float(), retained_probability),
            generator=generator,
        )
        scaled = tensor.float() * random_mask / max(retained_probability, 1e-8)
        stochastically_pruned.append(scaled)
    return linear_merge(stochastically_pruned, weights=weights)

