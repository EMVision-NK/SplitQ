"""Single-file calibration checkpoint containing unfused parameters, not base model weights."""

import hashlib
import json
from pathlib import Path
import torch
from .config import SplitQConfig
from .models.adapter import (
    architecture,
    bake_layer,
    model_config_digest,
    wrap_layer,
    inference_weight_dtype,
)

FORMAT = "splitq.single.v3"
CHECKPOINT_FILE = "checkpoint.pt"


def checkpoint_config(manifest):
    """Read the configuration saved with the checkpoint."""
    if manifest.get("format") != FORMAT:
        raise ValueError(f"Expected {FORMAT} checkpoint")
    values = dict(manifest["config"])
    values.pop("modality_bias_loss", None)
    return SplitQConfig(**values)


def learned_state(layer):
    params = {
        n
        for n, _ in layer.named_parameters()
        if "trans." in n or "clip_factor" in n or "gate_logits" in n
    }
    buffers = {
        n
        for n, _ in layer.named_buffers()
        if n.endswith(
            ("isolated_mask", "U_svd_shared", "sigma_svd_shared", "Vt_svd_shared")
        )
    }
    return {
        k: v.detach().cpu().clone()
        for k, v in layer.state_dict().items()
        if k in params | buffers
    }


def restore_state(layer, state):
    expected = learned_state(layer)
    missing, extra = expected.keys() - state.keys(), state.keys() - expected.keys()
    if missing or extra:
        raise ValueError(
            f"Incomplete/incompatible layer state: missing={sorted(missing)}, extra={sorted(extra)}"
        )
    with torch.no_grad():
        target = layer.state_dict()
        for key, value in state.items():
            if value.shape != expected[key].shape or value.dtype != expected[key].dtype:
                raise ValueError(
                    f"Invalid checkpoint tensor {key}: {value.shape}, {value.dtype}"
                )
            if value.is_floating_point() and not torch.isfinite(value).all():
                raise ValueError(f"Nonfinite checkpoint tensor: {key}")
            target[key].copy_(value.to(target[key].device))


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    temp.replace(path)


class CheckpointWriter:
    def __init__(self, output, config, bundle):
        self.path = Path(output)
        self.path.mkdir(parents=True, exist_ok=True)
        if (
            (self.path / "manifest.json").exists()
            or (self.path / CHECKPOINT_FILE).exists()
            or list(self.path.glob("layer_*.pt"))
        ):
            raise FileExistsError(f"Refusing to overwrite checkpoint: {self.path}")
        self._states = []
        self.manifest = {
            "format": FORMAT,
            "complete": False,
            "config": config.to_dict(),
            "architecture": architecture(bundle),
            "model_config_sha256": model_config_digest(config.model_path),
            "metrics": [],
        }
        write_json(self.path / "manifest.json", self.manifest)

    def save_layer(self, index, layer, metrics):
        if index != len(self._states):
            raise ValueError("Checkpoint layers must be saved in order")
        self._states.append(learned_state(layer))
        self.manifest["metrics"].append(metrics)
        write_json(self.path / "manifest.json", self.manifest)

    def finish(self):
        if len(self._states) != self.manifest["architecture"]["num_layers"]:
            raise ValueError("Cannot mark a partial checkpoint complete")
        temp = self.path / (CHECKPOINT_FILE + ".tmp")
        torch.save(self._states, temp)
        temp.replace(self.path / CHECKPOINT_FILE)
        self.manifest["checkpoint"] = {
            "file": CHECKPOINT_FILE,
            "sha256": digest(self.path / CHECKPOINT_FILE),
        }
        self.manifest["complete"] = True
        write_json(self.path / "manifest.json", self.manifest)
        self._states.clear()


def read_manifest(path):
    file = Path(path) / "manifest.json"
    if not file.exists():
        raise ValueError("Expected a checkpoint directory containing manifest.json")
    m = json.loads(file.read_text())
    if m.get("format") != FORMAT or not m.get("complete"):
        raise ValueError("Expected a complete splitq.single.v3 checkpoint")
    checkpoint_config(m)
    entry = m.get("checkpoint", {})
    if entry.get("file") != CHECKPOINT_FILE:
        raise ValueError("Invalid checkpoint filename")
    file = Path(path) / CHECKPOINT_FILE
    if not file.is_file() or digest(file) != entry.get("sha256"):
        raise ValueError(f"Missing or corrupt checkpoint: {file.name}")
    if len(m["metrics"]) != m["architecture"]["num_layers"]:
        raise ValueError("Checkpoint metric count mismatch")
    return m


def load_checkpoint(bundle, path, device="cuda:0"):
    manifest = read_manifest(path)
    if manifest["architecture"] != architecture(bundle):
        raise ValueError("Checkpoint architecture does not match the loaded model")
    config = checkpoint_config(manifest)
    states = torch.load(
        Path(path) / CHECKPOINT_FILE, map_location="cpu", weights_only=True
    )
    if (
        not isinstance(states, list)
        or len(states) != manifest["architecture"]["num_layers"]
    ):
        raise ValueError("Invalid checkpoint layer states")
    num_layers = manifest["architecture"]["num_layers"]
    for i in range(num_layers):
        print(f"Restoring SplitQ layer {i + 1}/{num_layers}", flush=True)
        state = states[i]
        layer = bundle.decoder.layers[i].to(device)
        wrap_layer(layer, config, bundle.decoder.config, initialize_svd=False)
        layer.self_attn.add_fq_trans(state["self_attn.input_isolated_mask"].to(device))
        layer.mlp.add_fq_trans(
            state["mlp.input_isolated_mask"].to(device),
            state["mlp.down_isolated_mask"].to(device),
        )
        layer.float().to(device)
        restore_state(layer, state)
        # Fuse FP32 parameters under BF16 autocast before narrowing main-path weights.
        # Keep FP32 biases/norms and post-GEMM additions aligned with calibration.
        base_dtype = inference_weight_dtype(config, device)
        bake_layer(layer, training_autocast=str(device).startswith("cuda"))
        from .quantization.linear import SplitQLinear

        for module in layer.modules():
            if isinstance(module, SplitQLinear):
                module.linear.weight.data = module.linear.weight.data.to(base_dtype)
        layer.cpu()
        states[i] = None
    bundle.model.to(device).eval()
    return manifest
