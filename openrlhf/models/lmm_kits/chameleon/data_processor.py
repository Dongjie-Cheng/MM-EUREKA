import json
from typing import Dict, List, Optional, Union

import torch
from qwen_vl_utils import process_vision_info

from ..base.data_processor import BaseDataProcessor


class ChameleonDataProcessor(BaseDataProcessor):
    def _format_messages(self, messages: Union[Dict, List[str], str]) -> List[Dict]:
        if isinstance(messages, list) and isinstance(messages[0], str):
            formated_messages = [json.loads(m) for m in messages]
        elif isinstance(messages, str):
            formated_messages = [json.loads(messages)]
        elif isinstance(messages, dict):
            formated_messages = [messages]
        else:
            raise ValueError("Invalid messages format, must be a list of strings or a string or a dict")
        return formated_messages

    def __call__(
        self,
        messages,
        max_length,
        padding=True,
        device=None,
        return_tensors="pt",
        add_special_tokens=False,
        truncation=True,
    ) -> Dict:
        messages = self._format_messages(messages)
        processor = self.processor
        texts = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)
        batch = processor(
            text=texts,
            images=image_inputs,
            padding=padding,
            max_length=max_length,
            add_special_tokens=add_special_tokens,
            truncation=truncation,
            return_tensors=return_tensors,
        )
        if device:
            return {k: v.to(device) for k, v in batch.items()}
        return {k: v for k, v in batch.items()}

    def make_input_batch(self, inputs: List[Dict]) -> Dict:
        batch = {}
        keys = set()
        for inp in inputs:
            keys.update(inp.keys())
        for key in keys:
            values = [inp.get(key) for inp in inputs]
            if all(v is None for v in values):
                batch[key] = None
                continue
            if all(isinstance(v, torch.Tensor) for v in values if v is not None):
                tensors = [v for v in values if v is not None]
                if all(t.shape == tensors[0].shape for t in tensors):
                    batch[key] = torch.stack(tensors, dim=0)
                else:
                    batch[key] = tensors
            else:
                batch[key] = values
        return batch

    def split_input_batch(self, batch: Dict) -> List[Dict]:
        batch_size = None
        for value in batch.values():
            if isinstance(value, torch.Tensor):
                batch_size = value.size(0)
                break
            if isinstance(value, list):
                batch_size = len(value)
                break
        if batch_size is None:
            return []
        batch_kwargs = [{} for _ in range(batch_size)]
        for key, value in batch.items():
            if value is None:
                for i in range(batch_size):
                    batch_kwargs[i][key] = None
                continue
            if isinstance(value, torch.Tensor):
                for i, v in enumerate(torch.unbind(value)):
                    batch_kwargs[i][key] = v
                continue
            if isinstance(value, list):
                if len(value) != batch_size:
                    raise ValueError(f"Unexpected list length for key {key}: {len(value)} != {batch_size}")
                for i, v in enumerate(value):
                    batch_kwargs[i][key] = v
                continue
            raise ValueError(f"Unsupported batch value type for key {key}: {type(value)}")
        return batch_kwargs

    def build_vllm_inputs(self, messages: Union[Dict, List[str], str], **kwargs) -> List[Dict]:
        message_items = messages if isinstance(messages, list) else [messages]
        prompts = self.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if isinstance(prompts, str):
            prompts = [prompts]
        images = [self.get_images_from_messages(m) for m in message_items]
        vllm_inputs = []
        for prompt, imgs in zip(prompts, images):
            vllm_input = {"prompt": prompt}
            if imgs:
                vllm_input["multi_modal_data"] = {"image": imgs}
            vllm_inputs.append(vllm_input)
        return vllm_inputs

    def get_vllm_mm_kwargs(self, **kwargs) -> Optional[Dict]:
        return None


DataProcessor = ChameleonDataProcessor

__all__ = ["DataProcessor"]
