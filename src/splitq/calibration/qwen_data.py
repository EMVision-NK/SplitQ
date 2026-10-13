# Adapted from the local BalanceQ / FlatQuant research code. See LICENSE.
import torch
import copy
import numpy as np


class CalibrationProcessor:
    def __init__(self, model, tokenizer, processor):
        self.model = model
        self.tokenizer = tokenizer
        self.processor = processor
        self.num_params = sum((p.numel() for p in self.model.parameters()))
        self.device_map = getattr(model, "hf_device_map", {})

    def convert_data_item(self, data_item):
        conversations = data_item["conversations"]
        for conv in conversations:
            if conv["from"] == "human":
                user_text = conv["value"]
                if "<image>" in user_text:
                    user_text = user_text.replace("<image>", "")
                if "\n" in user_text:
                    user_text = user_text.replace("\n", "")
            if conv["from"] == "gpt":
                asst_text = conv["value"]
        image_path = data_item["image"]
        item = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": user_text},
                ],
            },
            {"role": "assistant", "content": asst_text},
        ]
        return item

    def preprocess_data(self, images, data_item):
        new_data_item = self.convert_data_item(data_item)
        data_item_text = self.processor.apply_chat_template(
            new_data_item, tokenize=False, add_generation_prompt=False
        )
        data_dict = self.processor(
            text=data_item_text,
            images=images,
            videos=None,
            padding=True,
            return_tensors="pt",
        )
        for k, v in data_dict.items():
            if isinstance(v, torch.Tensor):
                data_dict[k] = torch.squeeze(v)
        labels = copy.deepcopy(data_dict["input_ids"])
        start_token_id = self.tokenizer.encode("<|im_start|>")[0]
        ans_start_indice = torch.where(labels == start_token_id)[0][-1].item()
        labels[:ans_start_indice] = -100
        data_dict["labels"] = labels
        return data_dict

    @torch.no_grad()
    def generate_input(self, data_samples):
        input_ids = data_samples["input_ids"].to(next(self.model.parameters()).device)
        attention_mask = data_samples["attention_mask"].to(
            next(self.model.parameters()).device
        )
        labels = data_samples["labels"].to(next(self.model.parameters()).device)
        pixel_values = data_samples["pixel_values"].to(
            next(self.model.parameters()).device
        )
        image_grid_thw = data_samples["image_grid_thw"].to(
            next(self.model.parameters()).device
        )
        inputs_embeds = self.model.model.language_model.embed_tokens(input_ids)
        vit = self.model.model.visual
        pixel_values = pixel_values.to(dtype=self.model.dtype)
        vit_out = vit(pixel_values, grid_thw=image_grid_thw)
        if isinstance(vit_out, tuple):
            image_embeds = vit_out[0]
        elif hasattr(vit_out, "last_hidden_state"):
            image_embeds = vit_out.last_hidden_state
        else:
            image_embeds = vit_out
        image_mask = (
            (input_ids == self.model.config.image_token_id)
            .unsqueeze(-1)
            .expand_as(inputs_embeds)
        )
        image_embeds = image_embeds.to(
            device=inputs_embeds.device, dtype=inputs_embeds.dtype
        )
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
        vision_mask = input_ids == self.model.config.image_token_id
        answer_mask = labels != -100
        prompt_inputs = {"inputs_embeds": inputs_embeds}
        prompt_kwargs = {
            "labels": labels,
            "attention_mask": attention_mask,
            "vision_mask": vision_mask,
            "caption_mask": answer_mask,
        }
        return (prompt_inputs, prompt_kwargs)

    def data_collator(self, instances):
        pad_id = self.tokenizer.pad_token_id
        IGNORE_INDEX = -100
        first = instances[0]
        batch = {}
        batch_lens = [feat["input_ids"].shape for feat in instances]
        max_item_length = max(batch_lens)[0]
        for idx in range(len(instances)):
            feat = instances[idx]
            temp_input_ids = torch.LongTensor([pad_id] * max_item_length)
            temp_input_ids[-feat["input_ids"].shape[0] :] = feat["input_ids"]
            feat["input_ids"] = temp_input_ids
            temp_labels = torch.LongTensor([IGNORE_INDEX] * max_item_length)
            temp_labels[-feat["labels"].shape[0] :] = feat["labels"]
            feat["labels"] = temp_labels
            feat["attention_mask"] = feat["input_ids"].ne(pad_id).int()
        if "label" in first and first["label"] is not None:
            label = (
                first["label"].item()
                if isinstance(first["label"], torch.Tensor)
                else first["label"]
            )
            dtype = torch.long if isinstance(label, int) else torch.float
            batch["labels"] = torch.tensor([f["label"] for f in instances], dtype=dtype)
        elif "label_ids" in first and first["label_ids"] is not None:
            if isinstance(first["label_ids"], torch.Tensor):
                batch["labels"] = torch.stack([f["label_ids"] for f in instances])
            else:
                dtype = (
                    torch.long
                    if isinstance(first["label_ids"][0], int)
                    else torch.float
                )
                batch["labels"] = torch.tensor(
                    [f["label_ids"] for f in instances], dtype=dtype
                )
        for k, v in first.items():
            if (
                k not in ("label", "pixel_values", "image_grid_thw")
                and v is not None
                and (not isinstance(v, str))
            ):
                if isinstance(v, torch.Tensor):
                    try:
                        batch[k] = torch.stack([f[k] for f in instances])
                    except RuntimeError:
                        if k in ["pixel_values", "image_grid_thw", "video_grid_thw"]:
                            batch[k] = torch.cat([f[k] for f in instances], dim=0)
                        else:
                            from torch.nn.utils.rnn import pad_sequence

                            pad_value = 0
                            if k == "input_ids":
                                pad_value = 151643
                            elif k == "labels":
                                pad_value = -100
                            batch[k] = pad_sequence(
                                [f[k] for f in instances],
                                batch_first=True,
                                padding_value=pad_value,
                            )
                elif isinstance(v, np.ndarray):
                    batch[k] = torch.tensor(np.stack([f[k] for f in instances]))
                else:
                    batch[k] = torch.tensor([f[k] for f in instances])
            if k in "pixel_values":
                if isinstance(v, torch.Tensor):
                    batch[k] = torch.concat([f[k] for f in instances])
                elif isinstance(v, np.ndarray):
                    batch[k] = torch.concat(np.stack([f[k] for f in instances]))
                else:
                    batch[k] = torch.concat([f[k] for f in instances])
            if k in "image_grid_thw":
                if isinstance(v, torch.Tensor):
                    batch[k] = torch.stack([f[k] for f in instances])
            if k in "sample_id":
                batch[k] = [f[k] for f in instances]
        return batch
