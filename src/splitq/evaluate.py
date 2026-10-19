"""Evaluate a complete SplitQ checkpoint with lmms-eval."""

import argparse
from dataclasses import replace
import importlib.metadata
from pathlib import Path
import sys
from .checkpoint import (
    checkpoint_config,
    load_checkpoint,
    model_config_digest,
    read_manifest,
    write_json,
)

TASKS = {
    "mmmu": "mmmu",
    "mmmu_val": "mmmu_val",
    "scienceqa": "scienceqa",
    "ocrbench": "ocrbench",
    "textvqa": "textvqa",
    "textvqa_val": "textvqa_val",
    "seedbench": "seedbench",
    "vizwiz": "vizwiz_vqa",
    "vizwiz_vqa": "vizwiz_vqa",
    "vizwiz_vqa_val": "vizwiz_vqa_val",
}


def evaluation_output_dir(checkpoint, tasks, output_dir=None):
    if output_dir is not None:
        return Path(output_dir)
    project = Path(__file__).resolve().parents[2]
    root = (project if (project / "pyproject.toml").is_file() else Path.cwd()) / "eval"
    return root / f"{Path(checkpoint).resolve().name}_{'_'.join(tasks)}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument(
        "--tasks",
        required=True,
        help="Comma-separated tasks; mmmu includes validation/test, mmmu_val evaluates validation only",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--output-dir",
        help="Results directory; defaults to the source project's eval directory or ./eval when installed",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--cache-root",
        help="Optional local Arrow snapshot directory; otherwise use native Hugging Face loading",
    )
    args = parser.parse_args()
    names = args.tasks.split(",")
    if any(name not in TASKS for name in names):
        parser.error(f"Supported tasks: {', '.join(TASKS)}")
    if args.limit is not None and args.limit <= 0:
        parser.error("limit must be positive")
    output = evaluation_output_dir(args.checkpoint, names, args.output_dir)
    args.output_dir = str(output)
    if output.exists() and any(output.iterdir()):
        parser.error(
            "output-dir must be absent or empty (existing results will not be overwritten)"
        )
    output.mkdir(parents=True, exist_ok=True)
    # Native aggregators such as OCRBench read the output path from the CLI namespace.
    args.output_path = str(output)
    manifest = read_manifest(args.checkpoint)
    if manifest["model_config_sha256"] != model_config_digest(args.model_path):
        raise ValueError(
            "Base model config differs from calibration; use the same pretrained checkpoint"
        )
    config = replace(
        checkpoint_config(manifest),
        model_path=args.model_path,
        device=args.device,
    )
    if args.cache_root:
        from .evaluation.cache import install_cache_loader

        install_cache_loader(args.cache_root)
    from .models.adapter import load_model
    from .calibration.trainer import seed_everything

    seed_everything(config.seed)
    bundle = load_model(config)
    load_checkpoint(bundle, args.checkpoint, args.device)
    if config.family == "qwen2_5_vl":
        from .evaluation.qwen import SplitQQwenEval as Evaluator

        model_name = "qwen2_vl"
    else:
        from .evaluation.llava import SplitQLlavaEval as Evaluator

        model_name = "llava"
    lm = Evaluator(bundle, args.device)
    from lmms_eval import evaluator
    from lmms_eval.tasks import TaskManager

    tasks = [TASKS[name] for name in names]
    manager = TaskManager("INFO", model_name=model_name)
    results = evaluator.simple_evaluate(
        model=model_name,
        lm=lm,
        tasks=tasks,
        task_manager=manager,
        batch_size=1,
        device=args.device,
        limit=args.limit,
        log_samples=True,
        bootstrap_iters=1000,
        random_seed=config.seed,
        numpy_random_seed=config.seed,
        torch_random_seed=config.seed,
        fewshot_random_seed=config.seed,
        cli_args=args,
    )
    samples = results.pop("samples", {})
    write_json(output / "metrics.json", results)
    for task, values in samples.items():
        write_json(output / f"samples_{task}.json", values)
    forbidden = [
        name
        for name in sys.modules
        if name == "flatquant"
        or name.startswith("flatquant.")
        or name == "qmllm"
        or name.startswith("qmllm.")
    ]
    if forbidden:
        raise RuntimeError(f"Unexpected legacy core imports: {forbidden}")
    write_json(
        output / "run.json",
        {
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "checkpoint_sha256": manifest["checkpoint"]["sha256"],
            "model_config_sha256": manifest["model_config_sha256"],
            "config": config.to_dict(),
            "tasks": tasks,
            "limit": args.limit,
            "cache_root": args.cache_root,
            "legacy_core_imports": forbidden,
            "versions": {
                n: importlib.metadata.version(n) for n in ("torch", "transformers")
            },
        },
    )
    print(f"Evaluation complete: {output / 'metrics.json'}")


if __name__ == "__main__":
    main()
