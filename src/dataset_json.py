import json
from typing import Dict, Any, List, Optional

import torch
from PIL import Image
import torchvision.transforms.functional as TF


class PairedDatasetJSON(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_path: str,
        split: str,
        height: int = 512,
        width: int = 512,
        tokenizer=None,
    ) -> None:
        super().__init__()
        with open(dataset_path, "r") as f:
            data_all: Dict[str, Dict[str, Any]] = json.load(f)
        assert split in data_all, f"Split '{split}' not found in {dataset_path}. Keys: {list(data_all.keys())}"
        self.data: Dict[str, Dict[str, Any]] = data_all[split]
        self.img_ids: List[str] = list(self.data.keys())
        self.image_size = (height, width)
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.img_ids)

    def _load_and_preprocess(self, path: str) -> torch.Tensor:
        img = Image.open(path).convert("RGB")
        t = TF.to_tensor(img)  # [0,1]
        t = TF.resize(t, self.image_size, antialias=True)
        t = TF.normalize(t, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # [-1,1]
        return t

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_id = self.img_ids[idx]
        entry = self.data[img_id]

        input_img_path: str = entry["image"]
        output_img_path: str = entry["target_image"]
        ref_img_path: Optional[str] = entry.get("ref_image")
        caption: str = entry.get("prompt", "")

        try:
            img_t = self._load_and_preprocess(input_img_path)
            output_t = self._load_and_preprocess(output_img_path)
        except Exception as e:
            # try next index on failure
            next_idx = (idx + 1) % len(self)
            return self.__getitem__(next_idx)

        if ref_img_path is not None:
            ref_t = self._load_and_preprocess(ref_img_path)
            img_t = torch.stack([img_t, ref_t], dim=0)  # [2, C, H, W]
            output_t = torch.stack([output_t, ref_t], dim=0)  # [2, C, H, W]
        else:
            img_t = img_t.unsqueeze(0)  # [1, C, H, W]
            output_t = output_t.unsqueeze(0)

        out: Dict[str, Any] = {
            "output_pixel_values": output_t,
            "conditioning_pixel_values": img_t,
            "caption": caption,
        }

        if self.tokenizer is not None and caption is not None:
            input_ids = self.tokenizer(
                caption,
                max_length=self.tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).input_ids
            out["input_ids"] = input_ids

        return out


