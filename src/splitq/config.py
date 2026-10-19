"""SplitQ calibration configuration."""

from dataclasses import asdict, dataclass, fields
import math
from pathlib import Path


@dataclass
class SplitQConfig:
    family: str = "qwen2_5_vl"
    model_path: str = ""
    data_path: str = ""
    image_folder: str = ""
    output_dir: str = ""
    w_bits: int = 4
    a_bits: int = 4
    cws: bool = True
    mac: bool = True
    cws_w_bits: int = 16
    cws_a_bits: int = 16
    mac_w_bits: int = 16
    mac_a_bits: int = 16
    cws_rank_ratio: float = 0.02
    mac_rank_ratio: float = 0.05
    qkv_isolation_ratio: float = 0.06
    up_gate_isolation_ratio: float = 0.08
    down_isolation_ratio: float = 0.04
    nsamples: int = 256
    epochs: int = 15
    cali_bsz: int = 4
    flat_lr: float = 0.005
    seed: int = 42
    device: str = "cuda:0"
    dtype: str = "bfloat16"

    def __post_init__(self):
        if self.family not in {"qwen2_5_vl", "llava_v15"}:
            raise ValueError("Supported families: qwen2_5_vl, llava_v15")
        for name in ("w_bits", "a_bits"):
            value = getattr(self, name)
            if type(value) is not int or not 2 <= value <= 16:
                raise ValueError(f"{name} must be an integer between 2 and 16")
        for name in ("cws_w_bits", "cws_a_bits", "mac_w_bits", "mac_a_bits"):
            if getattr(self, name) not in {4, 8, 16}:
                raise ValueError(f"{name} must be 4, 8 or 16")
        for name in (
            "qkv_isolation_ratio",
            "up_gate_isolation_ratio",
            "down_isolation_ratio",
            "cws_rank_ratio",
            "mac_rank_ratio",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 < value < 1:
                raise ValueError(f"{name} must be finite and in (0, 1)")
        for name in ("nsamples", "epochs", "cali_bsz"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not math.isfinite(self.flat_lr) or self.flat_lr <= 0:
            raise ValueError("flat_lr must be finite and positive")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if self.dtype not in {"float16", "bfloat16", "float32"}:
            raise ValueError("dtype must be float16, bfloat16 or float32")

    def to_dict(self):
        return asdict(self)


def load_config(path=None, overrides=None):
    values = {}
    if path:
        import yaml

        values = yaml.safe_load(Path(path).read_text()) or {}
        if not isinstance(values, dict):
            raise ValueError("Configuration must be a YAML mapping")
    values.update({k: v for k, v in (overrides or {}).items() if v is not None})
    unknown = set(values) - {f.name for f in fields(SplitQConfig)}
    if unknown:
        raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
    return SplitQConfig(**values)
