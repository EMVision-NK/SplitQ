from dataclasses import dataclass
from functools import lru_cache
import math

import torch
from ..quantization.utils import get_main_decompose_dim


@dataclass(frozen=True)
class BlockIsolationPlan:
    qkv: int
    up_gate: int
    down: int


@lru_cache(maxsize=None)
def _isolation_count(total, target):
    """Determine the isolated channel count."""
    if total < 10 or not math.isfinite(target) or not 0 < target < total:
        raise ValueError("Invalid asymmetric isolation dimensions or target")
    best = None
    for count in sorted(range(4, total - 5, 2), key=lambda k: (abs(k - target), k)):
        distance = abs(count - target)
        if best is not None and distance > best[0]:
            break
        try:
            left, right = get_main_decompose_dim(total - count)
        except ValueError:
            continue
        candidate = (distance, abs(right / left - 2.5), count)
        if best is None or candidate < best:
            best = candidate
    if best is None:
        raise ValueError(f"No asymmetric split for input width {total}")
    return best[2]


def resolve_isolation_plan(hidden, intermediate, args):
    """Resolve the isolation budget at each position."""
    targets = (
        hidden * args.qkv_isolation_ratio,
        hidden * args.up_gate_isolation_ratio,
        intermediate * args.down_isolation_ratio,
    )
    return BlockIsolationPlan(
        *(
            _isolation_count(total, target)
            for total, target in zip((hidden, hidden, intermediate), targets)
        )
    )


@torch.no_grad()
def _percentile_ranks(text, token_chunk=64):
    """Compute per-token channel percentile ranks; ties use the right rank."""
    tokens, channels = text.shape
    ranks = torch.empty((tokens, channels), dtype=torch.float32, device="cpu")
    for start in range(0, tokens, token_chunk):
        values = text[start : start + token_chunk].contiguous()
        sorted_values = values.sort(dim=1).values.contiguous()
        ranks[start : start + len(values)] = (
            torch.searchsorted(sorted_values, values, right=True).float() / channels
        ).cpu()
    return ranks


@torch.no_grad()
def _text_scores(
    text,
    token_chunk=64,
    channel_chunk=256,
    max_iter=300,
    tol=1e-4,
):
    """Cluster each channel's token ranks and score the occupied centers."""
    cluster_count = 3
    if text.ndim != 2 or text.shape[0] < cluster_count or text.shape[1] < 3:
        raise ValueError(
            f"Text scoring needs at least {cluster_count} tokens and three channels"
        )
    if max_iter < 1 or tol < 0:
        raise ValueError("Invalid k-means iteration limit or tolerance")
    channels = text.shape[1]
    ranks = _percentile_ranks(text, token_chunk)
    scores = torch.empty(channels, dtype=torch.float32, device=text.device)
    # Initialize cluster centers independently for each channel.
    first = torch.randint(len(text), (channels,))
    second = torch.randint(len(text) - 1, (channels,))
    second += (second >= first).long()
    third = torch.randint(len(text) - 2, (channels,))
    lower = torch.minimum(first, second)
    upper = torch.maximum(first, second)
    third += (third >= lower).long()
    third += (third >= upper).long()
    init_indices = torch.stack((first, second, third))
    initial_centers = ranks.gather(0, init_indices)
    for start in range(0, channels, channel_chunk):
        rank = ranks[:, start : start + channel_chunk].to(text.device)
        centers = initial_centers[:, start : start + rank.shape[1]].to(text.device)
        active = torch.ones(rank.shape[1], dtype=torch.bool, device=text.device)
        for _ in range(max_iter):
            labels = (rank[:, None, :] - centers[None, :, :]).square().argmin(dim=1)
            sums = torch.zeros_like(centers)
            counts = torch.zeros_like(centers)
            sums.scatter_add_(0, labels, rank)
            counts.scatter_add_(0, labels, torch.ones_like(rank))
            updated = torch.where(counts > 0, sums / counts.clamp_min(1), centers)
            # Relocate empty centers to the rank with the largest current error.
            if (counts == 0).any():
                residual = (rank - centers.gather(0, labels)).square()
                farthest = rank.gather(0, residual.argmax(dim=0, keepdim=True))
                updated = torch.where(counts > 0, updated, farthest.expand_as(updated))
            shift = (updated - centers).abs().amax(dim=0)
            centers = torch.where(active[None, :], updated, centers)
            active &= shift > tol
            if not active.any():
                break
        labels = (rank[:, None, :] - centers[None, :, :]).square().argmin(dim=1)
        counts = torch.zeros_like(centers)
        counts.scatter_add_(0, labels, torch.ones_like(rank))
        occupied = counts > 0
        mean_center = (centers * occupied).sum(dim=0) / occupied.sum(dim=0)
        scores[start : start + rank.shape[1]] = (
            (centers - mean_center).square() * occupied
        ).sum(dim=0) / occupied.sum(dim=0)
    return scores


@torch.no_grad()
def _guard_text_magnitude(
    candidates,
    scores,
    selected,
    channel_chunk=256,
):
    """Include significant outliers within the text budget."""
    remaining = torch.ones(len(scores), dtype=torch.bool, device=scores.device)
    remaining[selected] = False
    remaining_indices = remaining.nonzero(as_tuple=True)[0]
    cap = math.floor(0.10 * len(selected))
    if cap == 0 or not len(remaining_indices):
        return selected
    # Compute quantiles in channel chunks to bound temporary memory.
    magnitudes = torch.empty(
        len(remaining_indices), dtype=torch.float32, device=candidates.device
    )
    for start in range(0, len(remaining_indices), channel_chunk):
        columns = remaining_indices[start : start + channel_chunk]
        values = candidates[:, columns].float().abs()
        if not torch.isfinite(values).all():
            raise ValueError("Text magnitude guard needs finite activations")
        magnitudes[start : start + len(columns)] = torch.quantile(values, 0.99, dim=0)
    q25, q75 = torch.quantile(magnitudes, magnitudes.new_tensor([0.25, 0.75]))
    threshold = q75 + 3.0 * (q75 - q25)
    eligible = (magnitudes > threshold).nonzero(as_tuple=True)[0]
    amount = min(len(eligible), cap)
    if not amount:
        return selected
    # Use stable ordering to prefer lower channel indices on ties.
    order = magnitudes[eligible].argsort(descending=True, stable=True)
    added_positions = eligible[order[:amount]]
    added = remaining_indices[added_positions]
    # Replace the lowest-scoring channels first, preserving lower indices on ties.
    drop_order = selected.argsort(descending=True)
    drop_order = drop_order[scores[selected[drop_order]].argsort(stable=True)]
    removed = selected[drop_order[:amount]]
    retained = selected[~torch.isin(selected, removed)]
    return torch.cat((retained, added))


@torch.no_grad()
def select_mocd_channels(
    vision,
    text,
    count,
    device=None,
):
    """Select modality-specific vision and text channels."""
    if vision.ndim != 2 or text.ndim != 2 or vision.shape[1] != text.shape[1]:
        raise ValueError("Vision and text saliency must have the same channel width")
    if vision.shape[0] == 0 or text.shape[0] < 3:
        raise ValueError(
            "Calibration needs image tokens and at least three text tokens"
        )
    if not 2 <= count < vision.shape[1]:
        raise ValueError("Invalid isolation channel count")
    device = vision.device if device is None else device
    vision_sample = torch.randperm(len(vision), device=vision.device)[:5000]
    vision = vision[vision_sample].float().abs().to(device)
    text_sample = torch.randperm(len(text), device=text.device)[:5000]
    text = text[text_sample].float().abs().to(device)
    vision_count = count // 2
    text_count = count - vision_count
    vision_indices = vision.amax(dim=0).topk(vision_count).indices
    remaining = torch.ones(vision.shape[1], dtype=torch.bool, device=device)
    remaining[vision_indices] = False
    remaining_indices = remaining.nonzero(as_tuple=True)[0]
    candidates = text[:, remaining_indices]
    text_scores = _text_scores(candidates)
    selected = text_scores.topk(text_count).indices
    selected = _guard_text_magnitude(candidates, text_scores, selected)
    text_indices = remaining_indices[selected]
    mask = torch.zeros(vision.shape[1], dtype=torch.bool, device=device)
    mask[vision_indices] = True
    mask[text_indices] = True
    if int(mask.sum()) != count:
        raise ValueError("Clustered channel selection did not preserve the budget")
    return mask


class MOCDMixin:
    def get_index(self, vision_mask, text_mask=None):
        self.vision_mask = vision_mask
        self.text_mask = text_mask

    def _collect(self, x, suffix):
        flat = x.detach().reshape(-1, x.shape[-1])
        mask = self.vision_mask.reshape(-1).bool().to(x.device)
        if mask.numel() != len(flat):
            raise ValueError(
                "Vision mask length does not match calibration hidden states"
            )
        getattr(self, f"v_saliency_{suffix}_list").append(flat[mask].cpu())
        if self.text_mask is None:
            raise ValueError("Clustered selection requires a valid-text token mask")
        text_mask = self.text_mask.reshape(-1).bool().to(x.device)
        if text_mask.numel() != len(flat) or (text_mask & mask).any():
            raise ValueError("Invalid or overlapping vision/text token masks")
        getattr(self, f"t_saliency_{suffix}_list").append(flat[text_mask].cpu())

    def in_channel_marker(self, x):
        self._collect(x, "in")

    def down_channel_marker(self, x):
        self._collect(x, "down")

    def _select(self, suffix, count):
        device = next(self.parameters()).device
        vision = torch.cat(getattr(self, f"v_saliency_{suffix}_list")).float()
        text = torch.cat(getattr(self, f"t_saliency_{suffix}_list")).float()
        return select_mocd_channels(
            vision,
            text,
            count,
            device=device,
        )

    def finalize_outliers(self):
        is_mlp = hasattr(self, "down_proj")
        h = self.hidden_size
        intermediate = (
            self.intermediate_size if is_mlp else self.config.intermediate_size
        )
        plan = resolve_isolation_plan(h, intermediate, self.args)
        mask = self._select("in", plan.up_gate if is_mlp else plan.qkv)
        down_mask = None
        if is_mlp:
            down_mask = self._select("down", plan.down)
        self.add_fq_trans(mask, down_mask)
        for name in (
            "v_saliency_in_list",
            "t_saliency_in_list",
            "v_saliency_down_list",
            "t_saliency_down_list",
        ):
            if hasattr(self, name):
                getattr(self, name).clear()
