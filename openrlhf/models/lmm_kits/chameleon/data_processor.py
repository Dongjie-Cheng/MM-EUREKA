from typing import Dict, List

import torch
from qwen_vl_utils import process_vision_info

from ..base.data_processor import BaseDataProcessor


class ChameleonDataProcessor(BaseDataProcessor):
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
            images=image_inputs if image_inputs else None,
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
        # each element has no batch dimension
        batch = {}
        for inp in inputs:
            batch.update({k: None for k, v in inp.items() if v is not None})
        for k in batch.keys():
            if k in ["input_ids", "attention_mask"]:
                batch[k] = torch.stack([inp[k] for inp in inputs if k in inp], dim=0)
            elif k == "pixel_values":
                pixel_values = []
                for inp in inputs:
                    if k not in inp:
                        continue
                    value = inp[k]
                    if isinstance(value, torch.Tensor) and value.dim() == 3:
                        value = value.unsqueeze(0)
                    pixel_values.append(value)
                batch[k] = torch.cat(pixel_values, dim=0)
            else:
                raise ValueError(f"Unknown key {k} for ChameleonDataProcessor")
        return batch

    def split_input_batch(self, batch: Dict) -> List[Dict]:
        batch_size = len(batch["input_ids"])
        batch_kwargs = [{} for _ in range(batch_size)]
        keys = []
        for k, v in batch.items():
            if v is not None:
                keys.append(k)
            else:
                for i in range(batch_size):
                    batch_kwargs[i][k] = None

        for k in ["input_ids", "attention_mask"]:
            if k in keys:
                vals = batch[k]
                if isinstance(vals, torch.Tensor):
                    vals = torch.unbind(vals)
                assert batch_size == len(vals)
                for i, v in enumerate(vals):
                    batch_kwargs[i][k] = v

        if "pixel_values" in keys:
            pixel_values = batch["pixel_values"]
            if isinstance(pixel_values, torch.Tensor) and pixel_values.dim() == 3:
                pixel_values = pixel_values.unsqueeze(0)
            image_token_id = self.processor.tokenizer.convert_tokens_to_ids(self.processor.image_token)
            image_seq_length = self.processor.image_seq_length
            for i in range(batch_size):
                input_ids_i = batch_kwargs[i]["input_ids"]
                if not isinstance(input_ids_i, torch.Tensor):
                    input_ids_i = torch.tensor(input_ids_i)
                image_token_count = (input_ids_i == image_token_id).sum().item()
                if image_token_count == 0:
                    batch_kwargs[i]["pixel_values"] = None
                    continue
                if image_token_count % image_seq_length != 0:
                    raise ValueError("Image token count is not divisible by image sequence length.")
                img_num = image_token_count // image_seq_length
                pixel_values_i = pixel_values[:img_num]
                assert len(pixel_values_i) == img_num
                pixel_values = pixel_values[img_num:]
                batch_kwargs[i]["pixel_values"] = pixel_values_i
            assert len(pixel_values) == 0
        return batch_kwargs


DataProcessor = ChameleonDataProcessor

__all__ = ["DataProcessor"]
