"""Model loading, decoder adaptation and fusion."""

from dataclasses import dataclass
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
from types import MethodType
import torch


@dataclass
class ModelBundle:
    model: object
    tokenizer: object
    processor: object
    family: str
    max_length: int = 2048

    @property
    def decoder(self):
        return (
            self.model.model.language_model
            if self.family == "qwen2_5_vl"
            else self.model.model
        )


@torch.no_grad()
def _encode_llava_images(model, images):
    """Encode images individually with the native FP16 CLIP/projector path."""
    projector = model.get_model().mm_projector
    dtype = next(projector.parameters()).dtype
    with torch.autocast(device_type=images.device.type, enabled=False):
        return torch.cat(
            [
                projector(model.get_vision_tower()(image.to(dtype=dtype)))
                for image in images.split(1)
            ],
            dim=0,
        )


def configure_llava_precision(model, dtype):
    """Set the language-model dtype while keeping the vision tower/projector in FP16."""
    decoder = model.get_model()
    for module in (decoder.embed_tokens, decoder.layers, decoder.norm, model.lm_head):
        module.to(dtype=dtype)
    model.get_vision_tower().to(dtype=torch.float16)
    decoder.mm_projector.to(dtype=torch.float16)
    model.encode_images = MethodType(_encode_llava_images, model)


def load_model(config):
    path = Path(config.model_path)
    if path.is_dir():
        raw = json.loads((path / "config.json").read_text())
        if config.family == "llava_v15" and raw.get("architectures") != [
            "LlavaLlamaForCausalLM"
        ]:
            raise ValueError(
                "LLaVA requires native liuhaotian/llava-v1.5 checkpoints, not llava-hf"
            )
        if config.family == "qwen2_5_vl" and raw.get("model_type") != "qwen2_5_vl":
            raise ValueError(
                "Expected a Qwen2.5-VL checkpoint (Qwen3 is not supported)"
            )
    if config.family == "qwen2_5_vl":
        from transformers import (
            AutoProcessor,
            AutoTokenizer,
            Qwen2_5_VLForConditionalGeneration,
        )

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            config.model_path,
            torch_dtype=getattr(torch, config.dtype),
            attn_implementation="sdpa",
        )
        processor = AutoProcessor.from_pretrained(config.model_path)
        tokenizer = AutoTokenizer.from_pretrained(config.model_path)
        bundle = ModelBundle(model, tokenizer, processor, config.family)
    else:
        from llava.model.builder import load_pretrained_model
        from llava import conversation

        conversation.default_conversation = conversation.conv_templates["vicuna_v1"]
        # The native builder moves CLIP; use the requested GPU context.
        context = (
            torch.cuda.device(config.device)
            if str(config.device).startswith("cuda")
            else nullcontext()
        )
        with context:
            tokenizer, model, processor, length = load_pretrained_model(
                config.model_path,
                None,
                "llava-v1.5-7b",
                device_map="cpu",
                # Load the vision tower/projector directly in their native precision.
                torch_dtype="float16",
                attn_implementation="sdpa",
                multimodal=True,
            )
        configure_llava_precision(model, getattr(torch, config.dtype))
        model.tie_weights()
        bundle = ModelBundle(model, tokenizer, processor, config.family, length)
    bundle.model.eval().requires_grad_(False)
    return bundle


def architecture(bundle):
    cfg = bundle.decoder.config
    return {
        "family": bundle.family,
        "num_layers": len(bundle.decoder.layers),
        **{
            key: getattr(cfg, key)
            for key in (
                "hidden_size",
                "intermediate_size",
                "num_attention_heads",
                "num_key_value_heads",
            )
        },
    }


def model_config_digest(path):
    file = Path(path) / "config.json"
    return hashlib.sha256(file.read_bytes()).hexdigest() if file.exists() else None


def wrap_layer(layer, config, decoder_config, *, initialize_svd=True):
    from .mlp import SplitQMLP

    if config.family == "qwen2_5_vl":
        from .qwen_attention import SplitQQwenAttention as Attention
    else:
        from .llama_attention import SplitQLlamaAttention as Attention
    index = layer.self_attn.layer_idx
    layer.self_attn = Attention(config, layer.self_attn, initialize_svd=initialize_svd)
    layer.mlp = SplitQMLP(
        config, layer.mlp, index, config=decoder_config, initialize_svd=initialize_svd
    )
    return layer


@torch.no_grad()
def bake_layer(layer, training_autocast=False):
    """Fuse transforms once using the calibration quantization order."""
    if layer.self_attn._eval_mode:
        raise ValueError("Layer has already been reparameterized")
    device = next(layer.parameters()).device
    context = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if training_autocast and device.type == "cuda"
        else nullcontext()
    )
    with context:
        layer.self_attn.reparameterize()
        layer.mlp.reparameterize()
    # Keep diagonal activation scaling in place rather than folding it into BF16 Norm/Up weights.
    layer.eval()


def inference_weight_dtype(config, device):
    """Cast fused weights directly to the calibration GEMM dtype."""
    return (
        torch.bfloat16
        if str(device).startswith("cuda")
        else getattr(torch, config.dtype)
    )
