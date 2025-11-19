import gzip
import json
import logging
import os
import random
from typing import Dict, List

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate

from .transforms_carla_factory import create_carla_rgb_transform


_logger = logging.getLogger(__name__)


def lidar_to_histogram_features(lidar: np.ndarray, crop: int = 256) -> np.ndarray:
    """
    Convert LiDAR point cloud into a 3-channel BEV histogram feature map.

    This mirrors the behavior of lidar_to_histogram_features used in
    carla_dataset_llm, and can be applied on-the-fly to raw XYZ points.
    """

    def splat_points(point_cloud: np.ndarray) -> np.ndarray:
        pixels_per_meter = 8
        hist_max_per_pixel = 5
        x_meters_max = 14
        y_meters_max = 28
        xbins = np.linspace(
            -2 * x_meters_max,
            2 * x_meters_max + 1,
            2 * x_meters_max * pixels_per_meter + 1,
        )
        ybins = np.linspace(-y_meters_max, 0, y_meters_max * pixels_per_meter + 1)
        hist = np.histogramdd(point_cloud[..., :2], bins=(xbins, ybins))[0]
        hist[hist > hist_max_per_pixel] = hist_max_per_pixel
        overhead_splat = hist / hist_max_per_pixel
        return overhead_splat

    if lidar.shape[-1] < 3:
        raise ValueError("Expected lidar points with at least 3 dims (x, y, z).")

    below = lidar[lidar[..., 2] <= -2.0]
    above = lidar[lidar[..., 2] > -2.0]
    if below.size == 0:
        below_features = np.zeros((crop, crop), dtype=np.float32)
    else:
        below_features = splat_points(below)
    if above.size == 0:
        above_features = np.zeros((crop, crop), dtype=np.float32)
    else:
        above_features = splat_points(above)
    total_features = below_features + above_features
    features = np.stack([below_features, above_features, total_features], axis=-1)
    features = np.transpose(features, (2, 0, 1)).astype(np.float32)
    return features


class Bench2DriveChatB2DVQADataset(Dataset):
    """
    Bench2Drive + Chat-B2D VQA dataset.

    Each sample corresponds to a single frame in a Bench2Drive scenario,
    paired with a single QA turn from the corresponding Chat-B2D JSON file.

    Expected directory layout:
      sensor_root/
        <scenario_id>/
          camera/rgb_front/*.jpg
          camera/rgb_front_left/*.jpg
          camera/rgb_front_right/*.jpg
          lidar/*.laz           (currently unused; lidar is set to zeros)
          anno/*.json.gz        (optional, currently unused)
      language_root/  (e.g., Chat-B2D/chat-B2D/train or val)
        <scenario_id>/
          00010.json
          00045.json
          ...

    A Chat-B2D JSON file is a list of [human, gpt] message pairs.
    For training we randomly pick one QA pair per frame; for validation we
    take the first available pair.
    """

    def __init__(
        self,
        sensor_root: str,
        language_root: str,
        split: str = "train",
        is_training: bool = True,
        input_rgb_size: int = 224,
        input_multi_view_size: int = 112,
        input_lidar_size: int = 224,
    ) -> None:
        super().__init__()

        self.sensor_root = sensor_root
        self.language_root = language_root
        self.split = split
        self.is_training = is_training

        self.input_lidar_size = input_lidar_size

        self.rgb_transform = create_carla_rgb_transform(
            input_rgb_size,
            is_training=is_training,
            scale=None,
        )
        self.rgb_center_transform = create_carla_rgb_transform(
            128,
            scale=None,
            is_training=is_training,
            need_scale=False,
        )
        self.multi_view_transform = create_carla_rgb_transform(
            input_multi_view_size,
            scale=None,
            is_training=is_training,
        )

        self.samples: List[Dict] = []
        self._build_index()

        _logger.info(
            "Bench2DriveChatB2DVQADataset split=%s, samples=%d",
            split,
            len(self.samples),
        )

    def _build_index(self) -> None:
        """
        Build a flat list of (scenario, frame_id, json_path) triples by
        matching Chat-B2D JSON files to Bench2Drive sensor folders.
        """
        if not os.path.isdir(self.language_root):
            _logger.warning("language_root %s does not exist.", self.language_root)
            return

        scenario_names = sorted(
            d for d in os.listdir(self.language_root)
            if os.path.isdir(os.path.join(self.language_root, d))
        )

        for scenario in scenario_names:
            json_dir = os.path.join(self.language_root, scenario)
            sensor_dir = os.path.join(self.sensor_root, scenario)

            if not os.path.isdir(sensor_dir):
                _logger.warning(
                    "Sensor dir not found for scenario %s at %s, skipping.",
                    scenario,
                    sensor_dir,
                )
                continue

            for fname in sorted(os.listdir(json_dir)):
                if not fname.endswith(".json"):
                    continue
                frame_str = os.path.splitext(fname)[0]
                if not frame_str.isdigit():
                    continue

                # Require at least the three front cameras to exist.
                front = os.path.join(
                    sensor_dir, "camera", "rgb_front", f"{frame_str}.jpg"
                )
                front_left = os.path.join(
                    sensor_dir, "camera", "rgb_front_left", f"{frame_str}.jpg"
                )
                front_right = os.path.join(
                    sensor_dir, "camera", "rgb_front_right", f"{frame_str}.jpg"
                )

                if not (os.path.isfile(front) and os.path.isfile(front_left) and os.path.isfile(front_right)):
                    continue

                self.samples.append(
                    {
                        "scenario": scenario,
                        "frame_str": frame_str,
                        "json_path": os.path.join(json_dir, fname),
                    }
                )

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, path: str) -> Image.Image:
        img = Image.open(path).convert("RGB")
        return img

    def _load_qa_pair(self, json_path: str) -> Dict[str, str]:
        """
        Load a single QA pair from a Chat-B2D JSON file.
        The file format is:
          [
            [ {"from": "human", "value": ...}, {"from": "gpt", "value": ...} ],
            ...
          ]
        """
        with open(json_path, "r") as f:
            convo = json.load(f)

        pairs: List[Dict[str, str]] = []
        for turn in convo:
            if not isinstance(turn, list) or len(turn) < 2:
                continue
            human = None
            gpt = None
            for msg in turn:
                role = msg.get("from", "")
                if role == "human":
                    human = msg.get("value", "").strip()
                elif role == "gpt":
                    gpt = msg.get("value", "").strip()
            if human and gpt:
                pairs.append({"question": human, "answer": gpt})

        if not pairs:
            return {"question": "", "answer": ""}

        if self.is_training:
            chosen = random.choice(pairs)
        else:
            chosen = pairs[0]
        return chosen

    def _load_lidar_bev(self, sensor_dir: str, frame_str: str) -> torch.Tensor:
        """
        Load raw LiDAR from LAZ and convert to histogram BEV features on-the-fly.
        Falls back to zeros if laspy is unavailable or reading fails.
        """
        laz_path = os.path.join(sensor_dir, "lidar", f"{frame_str}.laz")
        bev_shape = (3, self.input_lidar_size, self.input_lidar_size)

        if not os.path.isfile(laz_path):
            _logger.warning("LiDAR file not found at %s, using zeros.", laz_path)
            return torch.zeros(bev_shape, dtype=torch.float32)

        try:
            import laspy
        except ImportError:
            _logger.warning("laspy is not installed; using zero LiDAR features.")
            return torch.zeros(bev_shape, dtype=torch.float32)

        try:
            with laspy.open(laz_path) as fh:
                las = fh.read()
            points = np.vstack([las.x, las.y, las.z]).T.astype(np.float32)
        except Exception as exc:
            _logger.warning("Failed to read LiDAR %s: %s", laz_path, exc)
            return torch.zeros(bev_shape, dtype=torch.float32)

        try:
            bev = lidar_to_histogram_features(points, crop=self.input_lidar_size)
        except Exception as exc:
            _logger.warning("Failed to convert LiDAR to BEV for %s: %s", laz_path, exc)
            return torch.zeros(bev_shape, dtype=torch.float32)

        return torch.from_numpy(bev)

    def __getitem__(self, idx: int) -> Dict:
        meta = self.samples[idx]
        scenario = meta["scenario"]
        frame_str = meta["frame_str"]

        sensor_dir = os.path.join(self.sensor_root, scenario)
        camera_dir = os.path.join(sensor_dir, "camera")

        front = self._load_image(
            os.path.join(camera_dir, "rgb_front", f"{frame_str}.jpg")
        )
        front_left = self._load_image(
            os.path.join(camera_dir, "rgb_front_left", f"{frame_str}.jpg")
        )
        front_right = self._load_image(
            os.path.join(camera_dir, "rgb_front_right", f"{frame_str}.jpg")
        )

        rgb = self.rgb_transform(front)
        rgb_center = self.rgb_center_transform(front)
        rgb_left = self.multi_view_transform(front_left)
        rgb_right = self.multi_view_transform(front_right)

        lidar = self._load_lidar_bev(sensor_dir, frame_str)

        # Measurements: 6-dim command one-hot + 1-dim speed.
        measurements = torch.zeros(7, dtype=torch.float32)
        anno_path = os.path.join(sensor_dir, "anno", f"{frame_str}.json.gz")
        if os.path.isfile(anno_path):
            try:
                with gzip.open(anno_path, "rt") as f:
                    anno = json.load(f)
                speed = float(anno.get("speed", 0.0))
                next_command = int(anno.get("next_command", 0))
                cmd_one_hot = [0.0] * 6
                cmd_idx = next_command - 1
                if cmd_idx < 0 or cmd_idx >= 6:
                    cmd_idx = 3
                cmd_one_hot[cmd_idx] = 1.0
                meas_np = np.array(cmd_one_hot + [speed], dtype=np.float32)
                measurements = torch.from_numpy(meas_np)
            except Exception as exc:
                _logger.warning(
                    "Failed to load measurements from %s: %s", anno_path, exc
                )

        qa = self._load_qa_pair(meta["json_path"])
        sample_id = f"{scenario}/{frame_str}"

        return {
            "id": sample_id,
            "rgb": rgb,
            "rgb_left": rgb_left,
            "rgb_right": rgb_right,
            "rgb_center": rgb_center,
            "lidar": lidar,
            "measurements": measurements,
            "vqa_question": qa["question"],
            "vqa_answer": qa["answer"],
        }

    def collater(self, samples: List[Dict]) -> Dict:
        return default_collate(samples)
