"""
Bench2Drive + Chat-B2D VQA dataset builder.
"""

from lavis.common.registry import registry
from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from lavis.datasets.datasets.bench2drive_chatb2d_vqa import Bench2DriveChatB2DVQADataset


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

        datasets = {}
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

        return datasets

