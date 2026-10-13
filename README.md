<div align="center">

## Breaking Modality Heterogeneity in Low-Bit Quantization for Large Vision-Language Models

Yi Zhong · Haotong Qin · Xindong Zhang · Lei Zhang · Guolei Sun

[![arXiv](https://img.shields.io/badge/arXiv-2605.19929-b31b1b.svg)](https://arxiv.org/abs/2605.19929)
[![Code](https://img.shields.io/badge/Code-GitHub-black.svg)](https://github.com/EMVision-NK/SplitQ)

</div>

PyTorch implementation of SplitQ for **Qwen2.5-VL** and **LLaVA-v1.5**.

SplitQ combines **Modality-specific Outlier Channel Decoupling (MOCD)** with **Adaptive Cross-modal Calibration (ACC)**, including **Cross-modal Weight Smoothing (CWS)** and **Modality-specific Activation Compensation (MAC)**.

## News

- **[2026-10-13]** We release the first version of the SplitQ code (**v1.0.0**).
- **[2026-09-25]** SplitQ has been accepted to **NeurIPS 2026**!
- **[2026-05-19]** The initial manuscript was submitted to [arXiv](https://arxiv.org/abs/2605.19929).

## Installation

Use Linux, Python 3.10 and a CUDA-enabled GPU. Qwen and LLaVA require separate environments because their Transformers versions differ. Run the commands below from the repository root.

### Qwen2.5-VL

```bash
conda create -n splitq_qwen python=3.10 pip -y
conda activate splitq_qwen
export PYTHONNOUSERSITE=1
unset PYTHONPATH

pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements/qwen.txt
pip install -r requirements/evaluation-runtime.txt
pip install --no-deps -r requirements/evaluation.txt
pip install --no-deps --no-build-isolation -e .
```

### LLaVA-v1.5

```bash
conda create -n splitq_llava python=3.10 pip -y
conda activate splitq_llava
export PYTHONNOUSERSITE=1
unset PYTHONPATH

pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements/llava.txt
pip install -r requirements/evaluation-runtime.txt
pip install --no-deps -r requirements/llava-source.txt
pip install --no-deps -r requirements/evaluation.txt
pip install --no-deps --no-build-isolation -e .
```

The source integrations are pinned to specific commits and installed with `--no-deps` to preserve the model-family dependencies.

## Calibration

### Prepare models and data

- Qwen: [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) and [Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct).
- LLaVA: [LLaVA-v1.5-7B](https://huggingface.co/liuhaotian/llava-v1.5-7b) and [LLaVA-v1.5-13B](https://huggingface.co/liuhaotian/llava-v1.5-13b).
- Calibration: [ShareGPT4V](https://github.com/ShareGPT4Omni/ShareGPT4V) conversations and the corresponding images. `--image-folder` must resolve the image paths in the JSON records.

Replace the paths below with your local files. The default configuration uses **W3A3**, 256 samples, 15 epochs and seed 42.

### Qwen2.5-VL

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/calibrate.sh qwen2_5_vl \
  --model-path /path/to/Qwen2.5-VL-7B-Instruct \
  --data-path /path/to/ShareGPT4V/sharegpt4v_coco.json \
  --image-folder /path/to/ShareGPT4V \
  --output-dir runs/qwen25vl_7b_w3a3
```

### LLaVA-v1.5

```bash
CUDA_VISIBLE_DEVICES=1 bash scripts/calibrate.sh llava_v15 \
  --model-path /path/to/llava-v1.5-7b \
  --data-path /path/to/ShareGPT4V/sharegpt4v_coco.json \
  --image-folder /path/to/ShareGPT4V \
  --output-dir runs/llava15_7b_w3a3
```

Model-family defaults are in `configs/`. Use `--w-bits` and `--a-bits` for any integer bit width from 2 to 16.

Calibration saves `manifest.json` and `checkpoint.pt`, without base model weights. Use a new output directory for each run. Evaluation requires a complete calibration checkpoint and the same base model.

## Evaluation

Use the environment matching the calibrated model family.

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/evaluate.sh \
  --checkpoint runs/qwen25vl_7b_w3a3 \
  --model-path /path/to/Qwen2.5-VL-7B-Instruct \
  --tasks scienceqa \
  --output-dir eval/qwen25vl_7b_w3a3_scienceqa
```

Tasks include `scienceqa`, `ocrbench`, `mmmu`, `textvqa`, `vizwiz_vqa` and `seedbench`. Multiple tasks can be comma-separated.

## Citation

```bibtex
@article{zhong2026breaking,
  title={Breaking Modality Heterogeneity in Low-Bit Quantization for Large Vision-Language Models},
  author={Zhong, Yi and Qin, Haotong and Zhang, Xindong and Zhang, Lei and Sun, Guolei},
  journal={arXiv preprint arXiv:2605.19929},
  year={2026}
}
```

## License

Original SplitQ contributions are released under the [MIT License](LICENSE). Third-party licenses and copyright notices are included in the same file.

## Acknowledgements

We thank [FlatQuant](https://github.com/ruikangliu/FlatQuant), [MBQ](https://github.com/thu-nics/MBQ), [LLaVA](https://github.com/haotian-liu/LLaVA) and [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval).
