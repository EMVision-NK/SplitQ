"""Read ShareGPT4V conversations without importing evaluation dependencies."""

import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image


def calibration_samples(config, bundle):
    if config.family == "qwen2_5_vl":
        from .qwen_data import CalibrationProcessor
    else:
        from .llava_data import CalibrationProcessor
    processor = CalibrationProcessor(bundle.model, bundle.tokenizer, bundle.processor)
    path = Path(config.data_path)
    if path.suffix == ".jsonl":
        records = [
            json.loads(line) for line in path.read_text().splitlines() if line.strip()
        ]
    else:
        records = json.loads(path.read_text())
    if not records:
        raise ValueError("Empty calibration dataset")
    indices = np.random.default_rng(config.seed).permutation(len(records))
    instances = []
    for i in range(config.nsamples):
        record = records[int(indices[i % len(indices)])]
        names = record.get("image")
        if not names:
            raise ValueError("SplitQ calibration requires an image in every sample")
        names = [names] if isinstance(names, str) else names
        images = []
        for name in names:
            with Image.open(Path(config.image_folder) / name) as img:
                images.append(img.convert("RGB"))
        instances.append(processor.preprocess_data(images, record))
    if config.family == "qwen2_5_vl":
        max_length = max(instance["input_ids"].shape[0] for instance in instances)
        pad_id = bundle.tokenizer.pad_token_id
        if pad_id is None:
            raise ValueError("Qwen2.5-VL calibration requires a pad token")
        for start in range(0, config.nsamples, config.cali_bsz):
            padded = []
            for instance in instances[start : start + config.cali_bsz]:
                width = max_length - instance["input_ids"].shape[0]
                item = dict(instance)
                item["input_ids"] = torch.nn.functional.pad(
                    instance["input_ids"], (width, 0), value=pad_id
                )
                item["labels"] = torch.nn.functional.pad(
                    instance["labels"], (width, 0), value=-100
                )
                padded.append(item)
            yield processor.generate_input(processor.data_collator(padded))
    else:
        yield from processor.calibration_batches(instances, config.cali_bsz)
