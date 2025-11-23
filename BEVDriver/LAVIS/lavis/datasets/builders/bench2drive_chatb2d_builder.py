"""
Bench2Drive + Chat-B2D VQA dataset builder.
"""

import random
from typing import Optional

import torch
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

        # Optional config to derive val_dev/val_test from val.
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

        for split in ann_info.keys():
            if split not in ["train", "val", "test", "val_dev", "val_test"]:
                continue

            # Handle derived splits from val
            if split in ("val_dev", "val_test") and val_split_cfg is not None:
                if val_dataset_cache is None:
                    base_val_cfg = ann_info.get("val", None)
                    if base_val_cfg is None:
                        continue
                    val_dataset_cache = Bench2DriveChatB2DVQADataset(
                        sensor_root=base_val_cfg.sensor_root,
                        language_root=base_val_cfg.language_root,
                        split="val",
                        is_training=False,
                        input_rgb_size=base_val_cfg.get("input_rgb_size", 224),
                        input_multi_view_size=base_val_cfg.get("input_multi_view_size", 112),
                        input_lidar_size=base_val_cfg.get("input_lidar_size", 224),
                    )

                total = len(val_dataset_cache)
                if total == 0:
                    datasets[split] = None
                    continue

                rng = random.Random(split_seed)
                indices = list(range(total))
                rng.shuffle(indices)

                if dev_size is not None:
                    dev_len = min(dev_size, total)
                elif dev_ratio is not None:
                    dev_len = max(1, int(total * float(dev_ratio)))
                else:
                    dev_len = total // 2

                dev_indices = indices[:dev_len]
                test_indices = indices[dev_len:]

                if split == "val_dev":
                    datasets[split] = SubsetWithCollater(val_dataset_cache, dev_indices)
                else:
                    datasets[split] = SubsetWithCollater(val_dataset_cache, test_indices)
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
                split=split if split in ("train", "val", "test") else "val",
                is_training=(split == "train"),
                input_rgb_size=input_rgb_size,
                input_multi_view_size=input_multi_view_size,
                input_lidar_size=input_lidar_size,
            )

        return datasets
