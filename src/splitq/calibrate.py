"""Calibration entry point: python -m splitq.calibrate --config configs/qwen2_5_vl.yaml ..."""

import argparse
from .config import load_config


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate SplitQ decoder layers on image-text conversations"
    )
    parser.add_argument("--config")
    for flag in (
        "family",
        "model-path",
        "data-path",
        "image-folder",
        "output-dir",
        "device",
        "dtype",
    ):
        parser.add_argument("--" + flag)
    for flag in (
        "qkv-isolation-ratio",
        "up-gate-isolation-ratio",
        "down-isolation-ratio",
    ):
        parser.add_argument("--" + flag, type=float)
    for flag in (
        "w-bits",
        "a-bits",
        "cws-w-bits",
        "cws-a-bits",
        "mac-w-bits",
        "mac-a-bits",
        "nsamples",
        "epochs",
        "cali-bsz",
        "seed",
    ):
        parser.add_argument("--" + flag, type=int)
    parser.add_argument("--flat-lr", type=float)
    parser.add_argument("--cws-rank-ratio", type=float)
    parser.add_argument("--mac-rank-ratio", type=float)
    for flag in (
        "cws",
        "mac",
    ):
        parser.add_argument(
            "--" + flag, action=argparse.BooleanOptionalAction, default=None
        )
    args = vars(parser.parse_args())
    config = load_config(args.pop("config"), args)
    if not all(
        (config.model_path, config.data_path, config.image_folder, config.output_dir)
    ):
        parser.error("model-path, data-path, image-folder and output-dir are required")
    from .models.adapter import load_model
    from .calibration.trainer import calibrate, seed_everything

    seed_everything(config.seed)
    manifest = calibrate(config, load_model(config))
    print(
        f"Complete: {manifest['architecture']['num_layers']} decoder layers saved to {config.output_dir}"
    )


if __name__ == "__main__":
    main()
