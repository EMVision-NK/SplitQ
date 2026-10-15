"""Connect a restored SplitQ model to native lmms-eval LLaVA generation."""

import torch
from lmms_eval.api.model import lmms
from lmms_eval.models.llava import Llava


class SplitQLlavaEval(Llava):
    def __init__(self, bundle, device):
        lmms.__init__(self)
        self._model, self._tokenizer = bundle.model, bundle.tokenizer
        self._image_processor = bundle.processor
        self._max_length = bundle.max_length
        self._config = bundle.model.config
        self._device = torch.device(device)
        self.device_map = device
        self.batch_size_per_gpu = 1
        self.truncation = True
        self.conv_template = "vicuna_v1"
        self.use_cache = True
        self.truncate_context = False
        self._rank, self._world_size = 0, 1

    def generate_until(self, requests):
        # Use the decoder precision used during calibration.
        with (
            torch.no_grad(),
            torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=self.device.type == "cuda",
            ),
        ):
            return super().generate_until(requests)

    def loglikelihood(self, requests):
        with (
            torch.no_grad(),
            torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=self.device.type == "cuda",
            ),
        ):
            return super().loglikelihood(requests)
