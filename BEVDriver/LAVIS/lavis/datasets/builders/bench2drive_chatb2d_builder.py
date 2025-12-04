"""
Bench2Drive + Chat-B2D VQA dataset builder.
"""

import json
import random
import re
from typing import Optional

from torch.utils.data import Subset

from lavis.common.registry import registry
from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from lavis.datasets.datasets.bench2drive_chatb2d_vqa import Bench2DriveChatB2DVQADataset


class SubsetWithCollater(Subset):
    """torch.utils.data.Subset that forwards the underlying collater if present."""

    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)
        self.collater = getattr(dataset, "collater", None)


@registry.register_builder("bench2drive_chatb2d")
class Bench2DriveChatB2DBuilder(BaseDatasetBuilder):
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/bench2drive_chatb2d/defaults.yaml",
    }

    def __init__(self, cfg=None):
        self.config = cfg
        # Matches coordinates like "<12.3, -4.56>"
        self._coord_pattern = re.compile(r"<-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?>")

    def build_datasets(self):
        return self.build()

    def build(self):
        build_info = self.config.build_info
        ann_info = build_info.annotations
        filter_coords = bool(getattr(self.config, "filter_coord_answers", False))

        # Optional config to derive val_dev / val_test from the provided val.
        val_split_cfg = self.config.get("val_split", None)
        dev_ratio: Optional[float] = None
        dev_size: Optional[int] = None
        split_seed: int = 42
        if val_split_cfg is not None:
            dev_ratio = val_split_cfg.get("dev_ratio", None)
            dev_size = val_split_cfg.get("dev_size", None)
            split_seed = int(val_split_cfg.get("seed", 42))

        datasets = {}
        val_dataset_cache = None

        # Build declared splits (train/val/test)
        for split in ann_info.keys():
            if split not in ["train", "val", "test"]:
                continue

            split_cfg = ann_info.get(split)
            sensor_root = split_cfg.sensor_root
            language_root = split_cfg.language_root

            input_rgb_size = split_cfg.get("input_rgb_size", 224)
            input_multi_view_size = split_cfg.get("input_multi_view_size", 112)
            input_lidar_size = split_cfg.get("input_lidar_size", 224)

            ds = Bench2DriveChatB2DVQADataset(
                sensor_root=sensor_root,
                language_root=language_root,
                split=split,
                is_training=(split == "train"),
                input_rgb_size=input_rgb_size,
                input_multi_view_size=input_multi_view_size,
                input_lidar_size=input_lidar_size,
            )
            if filter_coords and len(ds) > 0:
                keep_indices = []
                dropped = 0

                for idx, meta in enumerate(ds.samples):
                    try:
                        with open(meta["json_path"], "r") as f:
                            convo = json.load(f)
                    except Exception:
                        # If JSON is unreadable, drop it.
                        dropped += 1
                        continue

                    drop = False
                    for turn in convo:
                        if not isinstance(turn, list):
                            continue
                        for msg in turn:
                            if msg.get("from", "") == "gpt":
                                val = str(msg.get("value", ""))
                                if self._coord_pattern.search(val):
                                    drop = True
                                    break
                        if drop:
                            break
                    if drop:
                        dropped += 1
                    else:
                        keep_indices.append(idx)

                if dropped > 0:
                    ds = SubsetWithCollater(ds, keep_indices)
                    print(
                        f"[bench2drive_chatb2d] filtered {dropped} / {len(keep_indices)+dropped} samples with coordinate-like answers in split {split}."
                    )

            datasets[split] = ds

            if split == "val" and val_split_cfg is not None:
                val_dataset_cache = datasets[split]

        # Derive val_dev / val_test from val if configured and available.
        if val_dataset_cache is not None and len(val_dataset_cache) > 0:
            rng = random.Random(split_seed)
            indices = list(range(len(val_dataset_cache)))
            rng.shuffle(indices)

            if dev_size is not None:
                dev_len = min(dev_size, len(val_dataset_cache))
            elif dev_ratio is not None:
                dev_len = max(1, int(len(val_dataset_cache) * float(dev_ratio)))
            else:
                dev_len = len(val_dataset_cache) // 2

            dev_indices = indices[:dev_len]
            test_indices = indices[dev_len:]

            datasets["val_dev"] = SubsetWithCollater(val_dataset_cache, dev_indices)
            datasets["val_test"] = SubsetWithCollater(val_dataset_cache, test_indices)

        return datasets
