"""SplitQ layer-wise calibration."""

import gc
import math
import random
import numpy as np
import torch
from torch import nn
from .data import calibration_samples
from ..checkpoint import CheckpointWriter
from ..models.adapter import wrap_layer
from ..models.mocd import resolve_isolation_plan


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def map_tensors(value, fn):
    if torch.is_tensor(value):
        return fn(value)
    if isinstance(value, tuple):
        return tuple(map_tensors(v, fn) for v in value)
    if isinstance(value, list):
        return [map_tensors(v, fn) for v in value]
    if isinstance(value, dict):
        return {k: map_tensors(v, fn) for k, v in value.items()}
    return value


class CaptureComplete(Exception):
    """Stop input capture without swallowing model errors."""


def slice_captured_value(value, index, batch_size, key=None):
    if torch.is_tensor(value):
        if key in {"position_embeddings", "position_ids"} and value.ndim > 1:
            if value.shape[1] == batch_size:
                return value[:, index : index + 1]
        if key != "cache_position" and value.ndim and value.shape[0] == batch_size:
            return value[index : index + 1]
        return value
    if isinstance(value, tuple):
        return tuple(
            slice_captured_value(item, index, batch_size, key) for item in value
        )
    if isinstance(value, list):
        return [slice_captured_value(item, index, batch_size, key) for item in value]
    if isinstance(value, dict):
        return {
            name: slice_captured_value(item, index, batch_size, name)
            for name, item in value.items()
        }
    return value


@torch.no_grad()
def capture_inputs(config, bundle):
    samples = []
    bundle.model.to(config.device)
    current_mask = None
    current_caption = None
    current_text = None

    def capture(module, args, kwargs):
        hidden = args[0] if args else kwargs.pop("hidden_states")
        kwargs = {
            k: v
            for k, v in kwargs.items()
            if k not in {"past_key_values", "past_key_value"}
        }
        hidden = hidden.detach().cpu()
        kwargs = map_tensors(kwargs, lambda t: t.detach().cpu())
        vision = current_mask.detach().cpu()
        for index in range(hidden.shape[0]):
            samples.append(
                {
                    "hidden": hidden[index : index + 1],
                    "kwargs": slice_captured_value(kwargs, index, hidden.shape[0]),
                    "vision": vision[index : index + 1],
                    "caption": (
                        current_caption[index : index + 1].detach().cpu()
                        if current_caption is not None
                        else None
                    ),
                    "text": (
                        current_text[index : index + 1].detach().cpu()
                        if current_text is not None
                        else None
                    ),
                }
            )
        raise CaptureComplete

    handle = bundle.decoder.layers[0].register_forward_pre_hook(
        capture, with_kwargs=True
    )
    old_cache = bundle.decoder.config.use_cache
    bundle.decoder.config.use_cache = False
    try:
        for inputs, kwargs in calibration_samples(config, bundle):
            current_mask = kwargs["vision_mask"]
            current_caption = kwargs.get("caption_mask")
            current_text = kwargs["attention_mask"].bool() & ~current_mask.bool()
            try:
                bundle.decoder(
                    **inputs, attention_mask=kwargs["attention_mask"], use_cache=False
                )
            except CaptureComplete:
                pass
            else:
                raise RuntimeError("First decoder layer was not reached")
            del inputs, kwargs
        print("Captured calibration sample done!", flush=True)
    finally:
        handle.remove()
        bundle.decoder.config.use_cache = old_cache
        bundle.model.cpu()
        gc.collect()
        torch.cuda.empty_cache()
    return samples


def run_layer(layer, sample, device):
    x = sample["hidden"].to(device).float()
    kwargs = map_tensors(sample["kwargs"], lambda t: t.to(device))
    with torch.autocast(
        device_type="cuda", dtype=torch.bfloat16, enabled=str(device).startswith("cuda")
    ):
        output = layer(x, **kwargs)
    return output[0] if isinstance(output, tuple) else output


def stack_sample_values(values):
    """Reassemble captured decoder arguments along their batch dimensions."""
    first = values[0]
    if torch.is_tensor(first):
        if first.ndim and first.shape[0] == 1:
            return torch.cat(values, dim=0)
        if first.ndim > 1 and first.shape[1] == 1:
            # Qwen multimodal rotary embeddings have shape [3, batch, seq, dim].
            return torch.cat(values, dim=1)
        if all(torch.equal(first, value) for value in values[1:]):
            return first
        raise ValueError(
            f"Cannot combine decoder tensors of shape {tuple(first.shape)}"
        )
    if isinstance(first, tuple):
        return tuple(
            stack_sample_values([value[i] for value in values])
            for i in range(len(first))
        )
    if isinstance(first, list):
        return [
            stack_sample_values([value[i] for value in values])
            for i in range(len(first))
        ]
    if isinstance(first, dict):
        return {
            key: stack_sample_values([value[key] for value in values]) for key in first
        }
    if all(value == first for value in values[1:]):
        return first
    raise ValueError("Cannot combine nonidentical decoder arguments")


def run_batch(layer, batch, device):
    x = torch.cat([sample["hidden"] for sample in batch], dim=0).to(device).float()
    kwargs = stack_sample_values([sample["kwargs"] for sample in batch])
    kwargs = map_tensors(kwargs, lambda t: t.to(device))
    with torch.autocast(
        device_type="cuda", dtype=torch.bfloat16, enabled=str(device).startswith("cuda")
    ):
        output = layer(x, **kwargs)
    return output[0] if isinstance(output, tuple) else output


def uses_modality_loss(config):
    return config.w_bits == 3 and config.a_bits in {2, 3}


def modality_reconstruction_loss(parts, vision_weight=0.3, text_weight=0.7):
    """Weighted MSE over vision tokens and response-text tokens."""
    vision_error = None
    caption_error = None
    vision_count = 0
    caption_count = 0
    for output, target, sample in parts:
        vision = sample.get("vision")
        caption = sample.get("caption")
        if vision is None or caption is None:
            raise ValueError("W3A3/W3A2 calibration requires vision and caption masks")
        if vision.shape != output.shape[:2] or caption.shape != output.shape[:2]:
            raise ValueError(
                "Modality masks must match the batch and sequence dimensions"
            )
        error = (output.float() - target.float()).square()
        vision_count += int(vision.sum().item()) * error.shape[-1]
        caption_count += int(caption.sum().item()) * error.shape[-1]
        vision = vision.to(error.device).unsqueeze(-1)
        caption = caption.to(error.device).unsqueeze(-1)
        current_vision = (error * vision).sum()
        current_caption = (error * caption).sum()
        vision_error = (
            current_vision if vision_error is None else vision_error + current_vision
        )
        caption_error = (
            current_caption
            if caption_error is None
            else caption_error + current_caption
        )
    if not vision_count or not caption_count:
        raise ValueError(
            "W3A3/W3A2 calibration needs nonempty visual and response-text masks"
        )
    return (
        vision_weight * vision_error / vision_count
        + text_weight * caption_error / caption_count
    )


def parameter_groups(layer, config):
    layer.requires_grad_(False)
    groups = {}
    for name, param in layer.named_parameters():
        lr = None
        if "trans." in name:
            lr = config.flat_lr
        elif "clip_factor" in name:
            lr = config.flat_lr * 10
        elif "gate_logits_cws" in name and config.cws:
            lr = config.flat_lr * 0.2
        elif "gate_logits_mac" in name and config.mac:
            lr = config.flat_lr
        if lr is not None:
            param.requires_grad_(True)
            groups.setdefault(lr, []).append(param)
    return [{"params": params, "lr": lr} for lr, params in groups.items()]


def calibrate(config, bundle):
    seed_everything(config.seed)
    writer = CheckpointWriter(config.output_dir, config, bundle)
    samples = capture_inputs(config, bundle)
    decoder_config = bundle.decoder.config
    hidden = decoder_config.hidden_size
    intermediate = decoder_config.intermediate_size
    isolation = resolve_isolation_plan(hidden, intermediate, config)
    equivalent_percent = math.floor(
        100
        * (3 * isolation.qkv + 2 * isolation.up_gate + isolation.down)
        / (6 * hidden + intermediate)
    )
    for index, layer in enumerate(bundle.decoder.layers):
        print(
            f"Calibrating SplitQ layer {index + 1}/{len(bundle.decoder.layers)}",
            flush=True,
        )
        print(
            f"Equivalent isolated channels ≈{equivalent_percent}%; "
            f"CWS rank target={config.cws_rank_ratio if config.cws else 'off'}; "
            f"MAC rank target={config.mac_rank_ratio if config.mac else 'off'}",
            flush=True,
        )
        layer.to(config.device)
        wrap_layer(layer, config, bundle.decoder.config)
        layer.float().to(config.device).eval()
        for part in (layer.self_attn, layer.mlp):
            part._ori_mode = True
        targets = []
        with torch.no_grad():
            for sample in samples:
                for part in (layer.self_attn, layer.mlp):
                    part.get_index(
                        sample["vision"].to(config.device),
                        sample.get("text"),
                    )
                targets.append(run_layer(layer, sample, config.device).cpu())
            for part in (layer.self_attn, layer.mlp):
                part.finalize_outliers()
                part.to(config.device)
                part.init_diag_scale(0.3)
                part._ori_mode = False
        layer.to(config.device)
        optimizer = torch.optim.AdamW(parameter_groups(layer, config))
        steps = config.epochs * math.ceil(len(samples) / config.cali_bsz)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=steps, eta_min=config.flat_lr * 1e-3
        )
        tracked = next(p for group in optimizer.param_groups for p in group["params"])
        initial = tracked.detach().clone()
        metrics = {"layer": index, "steps": 0, "losses": [], "finite_gradients": True}
        for epoch in range(config.epochs):
            order = list(range(len(samples)))
            epoch_losses = []
            for start in range(0, len(order), config.cali_bsz):
                optimizer.zero_grad(set_to_none=True)
                batch = order[start : start + config.cali_bsz]
                selected = [samples[j] for j in batch]
                lengths = {sample["hidden"].shape[1] for sample in selected}
                if len(lengths) == 1:
                    quant_out = run_batch(layer, selected, config.device)
                    fp_hidden = torch.cat([targets[j] for j in batch], dim=0).to(
                        config.device
                    )
                    if uses_modality_loss(config):
                        if any(sample.get("caption") is None for sample in selected):
                            raise ValueError(
                                "W3A3/W3A2 calibration requires caption masks"
                            )
                        vision = torch.cat([sample["vision"] for sample in selected])
                        caption = torch.cat([sample["caption"] for sample in selected])
                        loss = modality_reconstruction_loss(
                            [
                                (
                                    quant_out,
                                    fp_hidden,
                                    {"vision": vision, "caption": caption},
                                )
                            ],
                            0.3,
                            0.7,
                        )
                    else:
                        loss = nn.functional.mse_loss(
                            quant_out.float(), fp_hidden.float()
                        )
                else:
                    parts = [
                        (
                            run_layer(layer, samples[j], config.device),
                            targets[j].to(config.device),
                            samples[j],
                        )
                        for j in batch
                    ]
                    if uses_modality_loss(config):
                        loss = modality_reconstruction_loss(
                            parts,
                            0.3,
                            0.7,
                        )
                    else:
                        losses = [
                            (
                                nn.functional.mse_loss(output.float(), target.float()),
                                target.numel(),
                            )
                            for output, target, _ in parts
                        ]
                        loss = sum(value * count for value, count in losses) / sum(
                            count for _, count in losses
                        )
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite loss in layer {index}")
                (loss / loss.detach().clamp_min(1e-12)).backward()
                grads = [
                    p.grad
                    for group in optimizer.param_groups
                    for p in group["params"]
                    if p.grad is not None
                ]
                if not grads or not all(torch.isfinite(g).all() for g in grads):
                    raise FloatingPointError(
                        f"Missing/nonfinite gradients in layer {index}"
                    )
                optimizer.step()
                scheduler.step()
                metrics["steps"] += 1
                epoch_losses.append(loss.item())
            # Epoch MSE is the sum of batch losses.
            metrics["losses"].append(sum(epoch_losses))
            print(
                f"layer={index} epoch={epoch} mse={metrics['losses'][-1]:.6g}",
                flush=True,
            )
        metrics["parameter_delta_max"] = (tracked.detach() - initial).abs().max().item()
        if (
            not math.isfinite(metrics["parameter_delta_max"])
            or metrics["parameter_delta_max"] == 0
        ):
            raise FloatingPointError(f"Optimizer did not update layer {index}")
        writer.save_layer(index, layer, metrics)
        for sample, target in zip(samples, targets):
            sample["hidden"] = target
        layer.requires_grad_(False).cpu().to(dtype=getattr(torch, config.dtype))
        del optimizer, scheduler, initial, tracked, loss, grads, targets
        gc.collect()
        torch.cuda.empty_cache()
    writer.finish()
    return writer.manifest
