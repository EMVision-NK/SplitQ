"""Load local Arrow snapshots without changing lmms tasks, prompts, splits or metrics."""

import os
from pathlib import Path

SNAPSHOTS = {
    ("lmms-lab/ScienceQA", "ScienceQA-FULL"): (
        "lmms-lab___science_qa/ScienceQA-FULL/0.0.0/69dd4d6b67373d38f96a5badd5d24d0eb5bcdc50",
        "science_qa",
    ),
    ("echo840/OCRBench", "default"): (
        "echo840___ocr_bench/default/0.0.0/92a54bd1384387c178d5a07140a2d85e0a3d12e1",
        "ocr_bench",
    ),
    ("lmms-lab/MMMU", "default"): (
        "lmms-lab___mmmu/default/0.0.0/364f2e2eb107b36e07ff4c5a15f5947a759cef47",
        "mmmu",
    ),
    ("lmms-lab/textvqa", "default"): (
        "lmms-lab___textvqa/default/0.0.0/9c0699cd19768ac5ab97568f6b3cbac4c0062884",
        "textvqa",
    ),
    ("lmms-lab/VizWiz-VQA", "default"): (
        "lmms-lab___viz_wiz-vqa/default/0.0.0/d428a2dae984f79cf1b9d99467dfa883e0c30686",
        "viz_wiz-vqa",
    ),
    ("lmms-lab/SEED-Bench", "default"): (
        "lmms-lab___seed-bench/default/0.0.0/74c4ea0cee96786739e4ccb97d227818a05ae752",
        "seed-bench",
    ),
}


def cache_root(root=None):
    if root:
        return Path(root).expanduser().resolve()
    if os.environ.get("HF_DATASETS_CACHE"):
        return Path(os.environ["HF_DATASETS_CACHE"]).expanduser().resolve()
    home = Path(
        os.environ.get(
            "HF_HOME",
            Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
            / "huggingface",
        )
    )
    return (home / "datasets").expanduser().resolve()


def install_cache_loader(root=None):
    """Explicitly load supported snapshots; missing snapshots fail without downloading."""
    import datasets

    directory_root = cache_root(root)
    original = datasets.load_dataset

    def load(path, name=None, *args, **kwargs):
        snapshot = SNAPSHOTS.get((path, name or "default"))
        if snapshot is None:
            return original(path, name, *args, **kwargs)
        if args:
            raise ValueError(
                "Arrow snapshot loading does not accept extra positional arguments"
            )
        relative, prefix = snapshot
        directory = directory_root / relative
        if not (directory / "dataset_info.json").is_file():
            raise FileNotFoundError(
                f"Missing Arrow snapshot: {directory}; check --cache-root"
            )
        info = datasets.DatasetInfo.from_directory(str(directory))
        result = datasets.DatasetDict()
        for split, metadata in info.splits.items():
            single = directory / f"{prefix}-{split}.arrow"
            files = (
                [single]
                if single.exists()
                else sorted(directory.glob(f"{prefix}-{split}-*-of-*.arrow"))
            )
            if not files:
                raise ValueError(f"Missing cached split: {path}/{split}")
            pieces = [
                datasets.Dataset.from_file(str(file), info=info) for file in files
            ]
            result[split] = (
                pieces[0] if len(pieces) == 1 else datasets.concatenate_datasets(pieces)
            )
            if len(result[split]) != metadata.num_examples:
                raise ValueError(f"Incomplete cached sample count: {path}/{split}")
        print(
            "Loading Arrow snapshot: ",
            path,
            {key: len(value) for key, value in result.items()},
            flush=True,
        )
        split = kwargs.get("split")
        if split is not None:
            if split not in result:
                raise ValueError(f"Arrow snapshots support full splits only: {split}")
            return result[split]
        return result

    datasets.load_dataset = load
    return original
