"""
Bench2Drive + Chat-B2D VQA dataset builder.
"""

import random
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

    def build_datasets(self):
        return self.build()

    def build(self):
        build_info = self.config.build_info
        ann_info = build_info.annotations

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

            datasets[split] = Bench2DriveChatB2DVQADataset(
                sensor_root=sensor_root,
                language_root=language_root,
                split=split,
                is_training=(split == "train"),
                input_rgb_size=input_rgb_size,
                input_multi_view_size=input_multi_view_size,
                input_lidar_size=input_lidar_size,
            )

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

