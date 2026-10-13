# Adapted from the local BalanceQ / FlatQuant research code. See LICENSE.
import base64
from io import BytesIO
from typing import List, Tuple
import torch
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm
from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    eval_logger.warning(
        "Failed to import qwen_vl_utils; Please install it via `pip install qwen-vl-utils`"
    )


class SplitQQwenEval(lmms):
    """Qwen2.5-VL generation through lmms-eval."""

    def __init__(self, bundle, device):
        super().__init__()
        self._model, self._tokenizer = bundle.model, bundle.tokenizer
        self.processor = bundle.processor
        self._config = bundle.model.config
        self._device = torch.device(device)
        self.device_map = device
        self.batch_size_per_gpu = 1
        self.use_cache = True
        self._rank, self._world_size = 0, 1
        self._max_length = bundle.max_length

    @property
    def config(self):
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Loglikelihood is not implemented for Qwen2_VL")

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            toks = self.tokenizer.encode(x[0])
            return (-len(toks), x[0])

        pbar = tqdm(
            total=len(requests), disable=self.rank != 0, desc="Model Responding"
        )
        re_ords = utils.Collator(
            [reg.args for reg in requests], _collate, grouping=True
        )
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        for chunk in chunks:
            (contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split) = zip(*chunk)
            task = task[0]
            split = split[0]
            visuals = [
                doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id
            ]
            # Preserve each question's image group, including multi-image MMMU questions.
            gen_kwargs = dict(all_gen_kwargs[0])
            until = [self.tokenizer.decode(self.eot_token_id)]
            if "until" in gen_kwargs:
                until = gen_kwargs.pop("until")
                if isinstance(until, str):
                    until = [until]
                elif not isinstance(until, list):
                    raise ValueError(
                        f"Expected `gen_kwargs['until']` to be of type Union[str,list] but got {type(until)}"
                    )
            if isinstance(contexts, tuple):
                contexts = list(contexts)
            for i in range(len(contexts)):
                if "<image>" in contexts[i]:
                    contexts[i] = contexts[i].replace("<image>", "")
            messages = []
            for i, context in enumerate(contexts):
                if "<image>" in context:
                    context = context.replace("<image>", "")
                message = [
                    {"role": "system", "content": "You are a helpful assistant."}
                ]
                if len(visuals) > 0:
                    visual = visuals[i] if i < len(visuals) else None
                    if isinstance(visual, str) and visual.endswith(
                        (".mp4", ".avi", ".mov")
                    ):
                        import decord

                        vr = decord.VideoReader(visual)
                        first_frame = vr[0].asnumpy()
                        (height, width) = first_frame.shape[:2]
                        max_pixels = height * width
                        message.append(
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "video",
                                        "video": visual,
                                        "max_pixels": max_pixels,
                                    },
                                    {"type": "text", "text": context},
                                ],
                            }
                        )
                    elif isinstance(visual, Image.Image):
                        base64_image = visual.convert("RGB")
                        buffer = BytesIO()
                        base64_image.save(buffer, format="JPEG")
                        base64_bytes = base64.b64encode(buffer.getvalue())
                        base64_string = base64_bytes.decode("utf-8")
                        message.append(
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "image",
                                        "image": f"data:image/jpeg;base64,{base64_string}",
                                    },
                                    {"type": "text", "text": context},
                                ],
                            }
                        )
                    elif isinstance(visual, (list, tuple)) and all(
                        (isinstance(v, Image.Image) for v in visual)
                    ):
                        image_content = []
                        for v in visual:
                            base64_image = v.convert("RGB")
                            buffer = BytesIO()
                            base64_image.save(buffer, format="JPEG")
                            base64_bytes = base64.b64encode(buffer.getvalue())
                            base64_string = base64_bytes.decode("utf-8")
                            image_content.append(
                                {
                                    "type": "image",
                                    "image": f"data:image/jpeg;base64,{base64_string}",
                                }
                            )
                        message.append(
                            {
                                "role": "user",
                                "content": image_content
                                + [{"type": "text", "text": context}],
                            }
                        )
                    else:
                        message.append(
                            {
                                "role": "user",
                                "content": [{"type": "text", "text": context}],
                            }
                        )
                else:
                    message.append(
                        {"role": "user", "content": [{"type": "text", "text": context}]}
                    )
                messages.append(message)
            texts = [
                self.processor.apply_chat_template(
                    msg, tokenize=False, add_generation_prompt=True
                )
                for msg in messages
            ]
            (image_inputs, video_inputs) = process_vision_info(messages)
            inputs = self.processor(
                text=texts,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            if self.device_map == "auto":
                inputs = inputs.to("cuda")
            else:
                inputs = inputs.to(self.device)
            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 128
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = None
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1
            pad_token_id = self.tokenizer.pad_token_id
            # Use BF16 autocast, matching calibration and fusion.
            with (
                torch.no_grad(),
                torch.autocast(
                    device_type=self.device.type,
                    dtype=torch.bfloat16,
                    enabled=self.device.type == "cuda",
                ),
            ):
                cont = self.model.generate(
                    **inputs,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=pad_token_id,
                    do_sample=True if gen_kwargs["temperature"] > 0 else False,
                    temperature=gen_kwargs["temperature"],
                    top_p=gen_kwargs["top_p"],
                    num_beams=gen_kwargs["num_beams"],
                    max_new_tokens=gen_kwargs["max_new_tokens"],
                    use_cache=self.use_cache,
                )
            generated_ids_trimmed = [
                out_ids[len(in_ids) :]
                for (in_ids, out_ids) in zip(inputs.input_ids, cont)
            ]
            answers = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            for i, ans in enumerate(answers):
                for term in until:
                    if len(term) > 0:
                        ans = ans.split(term)[0]
                answers[i] = ans
            for ans, context in zip(answers, contexts):
                res.append(ans)
                self.cache_hook.add_partial(
                    "generate_until", (context, gen_kwargs), ans
                )
                pbar.update(1)
        res = re_ords.get_original(res)
        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation")
