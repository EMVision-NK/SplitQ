# Adapted from the local BalanceQ / FlatQuant research code. See LICENSE.
import torch
import copy
import re
from PIL import Image
from typing import Dict, Sequence
from llava import conversation as conversation_lib
from llava.constants import (
    IGNORE_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from .llava_tokens import preprocess_v1


class CalibrationProcessor:
    def __init__(self, model, tokenizer, processor=None):
        self.model = model
        self.tokenizer = tokenizer
        self.num_params = sum((p.numel() for p in self.model.parameters()))
        self.device_map = getattr(model, "hf_device_map", {})
        self.vision_tower = model.get_vision_tower()
        self.image_aspect_ratio = "pad"
        self.image_processor = self.vision_tower.image_processor
        self.is_multimodal = True
        self.mm_use_im_start_end = model.config.mm_use_im_start_end
        self.image_grid_pinpoints = getattr(model.config, "image_grid_pinpoints", None)
        self.image_crop_resolution = getattr(
            model.config, "image_crop_resolution", None
        )
        self.image_split_resolution = getattr(
            model.config, "image_split_resolution", None
        )

    def vision_preprocess(self, image: torch.FloatTensor):
        image_size = image.size
        image_aspect_ratio = self.image_aspect_ratio
        if image_aspect_ratio == "pad":

            def expand2square(pil_img, background_color):
                (width, height) = pil_img.size
                if width == height:
                    return pil_img
                elif width > height:
                    result = Image.new(pil_img.mode, (width, width), background_color)
                    result.paste(pil_img, (0, (width - height) // 2))
                    return result
                else:
                    result = Image.new(pil_img.mode, (height, height), background_color)
                    result.paste(pil_img, ((height - width) // 2, 0))
                    return result

            image = expand2square(
                image, tuple((int(x * 255) for x in self.image_processor.image_mean))
            )
            image = self.image_processor.preprocess(image, return_tensors="pt")[
                "pixel_values"
            ][0]
        else:
            image = self.image_processor.preprocess(image, return_tensors="pt")[
                "pixel_values"
            ][0]
        return (image, image_size, "image")

    def preprocess_multimodal(self, sources: Sequence[str]) -> Dict:
        is_multimodal = self.is_multimodal
        if not is_multimodal:
            return sources
        for source in sources:
            for sentence in source:
                num_im = len(re.findall(DEFAULT_IMAGE_TOKEN, sentence["value"]))
                if (
                    num_im == 1
                    and DEFAULT_IMAGE_TOKEN in sentence["value"]
                    and (not sentence["value"].startswith(DEFAULT_IMAGE_TOKEN))
                ):
                    sentence["value"] = (
                        sentence["value"].replace(DEFAULT_IMAGE_TOKEN, "").strip()
                    )
                    sentence["value"] = DEFAULT_IMAGE_TOKEN + "\n" + sentence["value"]
                    sentence["value"] = sentence["value"].strip()
                    if "mmtag" in conversation_lib.default_conversation.version:
                        sentence["value"] = sentence["value"].replace(
                            DEFAULT_IMAGE_TOKEN,
                            "<Image>" + DEFAULT_IMAGE_TOKEN + "</Image>",
                        )
                replace_token = DEFAULT_IMAGE_TOKEN
                if self.mm_use_im_start_end:
                    replace_token = (
                        DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
                    )
                sentence["value"] = sentence["value"].replace(
                    DEFAULT_IMAGE_TOKEN, replace_token
                )
                sentence["value"] = sentence["value"].replace(
                    "QA_GT_caption_based_noisy", ""
                )
        return sources

    def preprocess_data(self, images, data_item):
        sources = [data_item]
        if images is not None:
            if len(images) == 1:
                image = [self.vision_preprocess(images[0])]
            else:
                image = [self.vision_preprocess(f) for f in images]
                image = [[im[0], im[1], "image"] for im in image]
            sources = self.preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources])
            )
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])
        has_image = "image" in data_item
        data_dict = preprocess_v1(sources, self.tokenizer, has_image=has_image)
        if "prompt" in data_dict:
            prompt = data_dict["prompt"]
        else:
            prompt = None
        data_dict = dict(
            input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0]
        )
        if "image" in data_item:
            data_dict["image"] = image
        elif self.is_multimodal:
            crop_size = self.image_processor.crop_size
            data_dict["image"] = [
                (
                    torch.zeros(1, 3, crop_size["height"], crop_size["width"]),
                    (crop_size["width"], crop_size["height"]),
                    "text",
                )
            ]
        if prompt is not None:
            data_dict["prompt"] = prompt
        if "id" in data_item:
            data_dict["id"] = data_item["id"]
        return data_dict

    @torch.no_grad()
    def generate_input(self, data_samples):
        data_samples["input_ids"] = data_samples["input_ids"].to(
            next(self.model.parameters()).device
        )
        data_samples["attention_mask"] = data_samples["attention_mask"].to(
            next(self.model.parameters()).device
        )
        data_samples["labels"] = data_samples["labels"].to(
            next(self.model.parameters()).device
        )
        for i, _ in enumerate(data_samples["images"]):
            data_samples["images"][i] = data_samples["images"][i].to(
                device=next(self.model.parameters()).device,
                dtype=next(self.model.get_model().mm_projector.parameters()).dtype,
            )
        (
            input_ids,
            position_ids,
            attention_mask,
            past_key_values,
            input_embeds,
            labels,
        ) = self.model.prepare_inputs_labels_for_multimodal(
            data_samples["input_ids"],
            None,
            data_samples["attention_mask"],
            None,
            data_samples["labels"],
            data_samples["images"],
            data_samples["modalities"],
            data_samples["image_sizes"],
        )
        vision_sel = []
        for batch_idx, pre_input_ids in enumerate(data_samples["input_ids"]):
            # Locate image tokens after native image expansion, truncation and padding.
            valid_ids = pre_input_ids[data_samples["attention_mask"][batch_idx].bool()]
            image_positions = (valid_ids == IMAGE_TOKEN_INDEX).nonzero(as_tuple=True)[0]
            if not len(image_positions):
                raise ValueError(
                    "LLaVA calibration requires an image placeholder in every sample"
                )
            image_emb_len = self.vision_tower.num_patches
            # In cls_patch mode, native num_patches already includes CLS.
            if self.vision_tower.select_feature not in {"patch", "cls_patch"}:
                raise ValueError(
                    "LLaVA-v1.5 calibration supports patch/cls_patch visual features"
                )
            valid_positions = attention_mask[batch_idx].bool().nonzero(as_tuple=True)[0]
            if not len(valid_positions):
                raise ValueError("Empty multimodal calibration sample")
            offset, post_len = int(valid_positions[0]), len(valid_positions)
            cur_vision_sel = torch.zeros_like(
                attention_mask[batch_idx], dtype=torch.bool
            )
            for index, position in enumerate(image_positions.tolist()):
                start = position + (image_emb_len - 1) * index
                end = min(start + image_emb_len, post_len)
                if start < end:
                    cur_vision_sel[offset + start : offset + end] = True
            vision_sel.append(cur_vision_sel)
        vision_sel = torch.stack(vision_sel)
        vision_mask = vision_sel
        answer_mask = (labels != IGNORE_INDEX) & attention_mask.bool() & ~vision_mask
        prompt_inputs = {"inputs_embeds": input_embeds}
        prompt_kwargs = {
            "labels": labels,
            "attention_mask": attention_mask,
            "vision_mask": vision_mask,
            "caption_mask": answer_mask,
        }
        return (prompt_inputs, prompt_kwargs)

    def calibration_batches(self, instances, batch_size):
        """Encode image minibatches and assemble padded embeddings on CPU."""
        if not instances or batch_size < 1:
            raise ValueError(
                "Use nonempty calibration instances and a positive batch size"
            )
        device = next(self.model.parameters()).device
        batches = []
        for start in range(0, len(instances), batch_size):
            inputs, kwargs = self.generate_input(
                self.data_collator(instances[start : start + batch_size])
            )
            batches.append(
                (
                    {key: value.detach().cpu() for key, value in inputs.items()},
                    {key: value.detach().cpu() for key, value in kwargs.items()},
                )
            )
            del inputs, kwargs
        max_length = max(inputs["inputs_embeds"].shape[1] for inputs, _ in batches)
        left = getattr(self.model.config, "tokenizer_padding_side", "right") == "left"
        for inputs, kwargs in batches:
            width = max_length - inputs["inputs_embeds"].shape[1]
            padding = (width, 0) if left else (0, width)
            inputs = {
                "inputs_embeds": torch.nn.functional.pad(
                    inputs["inputs_embeds"],
                    (0, 0, *padding),
                    value=0,
                ).to(device)
            }
            kwargs = {
                key: torch.nn.functional.pad(
                    value,
                    padding,
                    value=IGNORE_INDEX if key == "labels" else 0,
                ).to(device)
                for key, value in kwargs.items()
            }
            yield inputs, kwargs

    def pad_sequence(self, input_ids, batch_first, padding_value):
        if self.tokenizer.padding_side == "left":
            input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=batch_first, padding_value=padding_value
        )
        if self.tokenizer.padding_side == "left":
            input_ids = torch.flip(input_ids, [1])
        return input_ids

    def data_collator(self, instances):
        (input_ids, labels) = tuple(
            (
                [instance[key] for instance in instances]
                for key in ("input_ids", "labels")
            )
        )
        input_ids = [
            _input_ids[: self.tokenizer.model_max_length] for _input_ids in input_ids
        ]
        labels = [_labels[: self.tokenizer.model_max_length] for _labels in labels]
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = 0
        input_ids = self.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = self.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        batch = dict(
            input_ids=input_ids,
            labels=labels.long() if labels.dtype == torch.int32 else labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
        if "image" in instances[0]:
            images = [instance["image"] for instance in instances]
            batch["image_sizes"] = [im[1] for im_list in images for im in im_list]
            batch["modalities"] = [im[2] for im_list in images for im in im_list]
            images = [im[0] for im_list in images for im in im_list]
            batch["images"] = images
        if "prompt" in instances[0]:
            batch["prompts"] = [instance["prompt"] for instance in instances]
        if "id" in instances[0]:
            batch["sample_id"] = [instance["id"] for instance in instances]
        return batch
