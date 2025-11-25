"""
Evaluate a trained BEV-QA model on a validation split and report SPICE (primary),
optionally BLEU-4 and CIDEr, while also saving per-example predictions for
qualitative analysis.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader

# Ensure bundled LAVIS package is on the path (repo-local).
REPO_ROOT = Path(__file__).resolve().parent
LAVIS_ROOT = REPO_ROOT / "BEVDriver" / "LAVIS"
if str(LAVIS_ROOT) not in sys.path:
    sys.path.insert(0, str(LAVIS_ROOT))

import lavis.tasks as tasks
from lavis.common.config import Config
from lavis.datasets.data_utils import prepare_sample
from lavis.tasks import *  # noqa: F401,F403 - needed for registry side effects
from lavis.datasets.builders import *  # noqa: F401,F403
from lavis.models import *  # noqa: F401,F403


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate BEV-QA model with SPICE.")
    parser.add_argument(
        "--cfg-path",
        default="BEVDriver/LAVIS/lavis/projects/bevqa/train.yaml",
        help="Path to config YAML (e.g., lavis/projects/bevqa/train.yaml)",
    )
    parser.add_argument(
        "--ckpt-path",
        required=True,
        help="Checkpoint path for the trained BEV-QA model.",
    )
    parser.add_argument(
        "--split",
        default="val_dev",
        help="Dataset split to evaluate (e.g., val, val_dev, val_test).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Evaluation batch size.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of dataloader workers.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device to run on (cuda or cpu).",
    )
    parser.add_argument(
        "--output",
        default="eval_results.json",
        help="Path to save qualitative results.",
    )
    parser.add_argument(
        "--options",
        nargs="+",
        help="Optional config overrides in key=value format (same as LAVIS runner).",
    )
    return parser.parse_args()


def build_dataloader(task, cfg: Config, split: str, batch_size: int, num_workers: int):
    datasets = task.build_datasets(cfg)
    # Use the first declared dataset (e.g., bench2drive_chatb2d)
    dataset_name = next(iter(datasets.keys()))
    if split not in datasets[dataset_name]:
        raise ValueError(f"Split '{split}' not found in dataset '{dataset_name}'. "
                         f"Available splits: {list(datasets[dataset_name].keys())}")
    dataset = datasets[dataset_name][split]

    collate_fn = getattr(dataset, "collater", None)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    return dataloader


def move_to_device(samples: Dict, device: torch.device) -> Dict:
    # Use existing helper to move tensors; keeps strings/lists intact.
    return prepare_sample(samples, cuda_enabled=device.type == "cuda")


def compute_metrics(gts: Dict, hyps: Dict) -> Dict[str, float]:
    from pycocoevalcap.spice.spice import Spice
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.cider.cider import Cider

    metrics = {}

    spice_scorer = Spice()
    spice_score, _ = spice_scorer.compute_score(gts, hyps)
    metrics["SPICE"] = spice_score

    bleu_scorer = Bleu(4)
    bleu_scores, _ = bleu_scorer.compute_score(gts, hyps)
    metrics["BLEU-4"] = bleu_scores[3]  # index 3 corresponds to BLEU-4

    cider_scorer = Cider()
    cider_score, _ = cider_scorer.compute_score(gts, hyps)
    metrics["CIDEr"] = cider_score

    return metrics


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # Robust device selection: fall back to CPU if CUDA is not available/healthy.
    if args.device.startswith("cuda"):
        try:
            use_cuda = torch.cuda.is_available()
        except Exception as exc:  # pragma: no cover - defensive
            logging.warning("CUDA availability check failed (%s); falling back to CPU.", exc)
            use_cuda = False
    else:
        use_cuda = False
    device = torch.device("cuda" if use_cuda else "cpu")

    # Resolve config path robustly.
    cfg_candidates = [
        Path(args.cfg_path),
        REPO_ROOT / args.cfg_path,
        LAVIS_ROOT / args.cfg_path,
    ]
    resolved_cfg = next((p for p in cfg_candidates if p.is_file()), None)
    if resolved_cfg is None:
        raise FileNotFoundError(f"Config not found. Tried: {cfg_candidates}")
    args.cfg_path = str(resolved_cfg)

    # Resolve checkpoint path robustly.
    ckpt_candidates = [
        Path(args.ckpt_path),
        REPO_ROOT / args.ckpt_path,
        LAVIS_ROOT / args.ckpt_path,
    ]
    resolved_ckpt = next((p for p in ckpt_candidates if p.is_file()), None)
    if resolved_ckpt is None:
        raise FileNotFoundError(f"Checkpoint not found. Tried: {ckpt_candidates}")
    args.ckpt_path = str(resolved_ckpt)

    cfg = Config(args)
    task = tasks.setup_task(cfg)
    model = task.build_model(cfg)

    # Load checkpoint weights.
    logging.info("Loading checkpoint from %s", args.ckpt_path)
    ckpt = torch.load(args.ckpt_path, map_location="cpu")
    state_dict = ckpt.get("model", ckpt)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        logging.warning("Missing keys when loading checkpoint: %s", ", ".join(missing))
    if unexpected:
        logging.warning("Unexpected keys when loading checkpoint: %s", ", ".join(unexpected))

    model.to(device)
    model.eval()

    dataloader = build_dataloader(
        task,
        cfg=cfg,
        split=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    records: List[Dict] = []
    gts: Dict[str, List[str]] = {}
    hyps: Dict[str, List[str]] = {}

    with torch.no_grad():
        for samples in dataloader:
            samples = move_to_device(samples, device)

            preds = model.generate(samples)
            ids = samples["id"]
            questions = samples["vqa_question"]
            refs = samples["vqa_answer"]

            for sid, q, ref, pred in zip(ids, questions, refs, preds):
                sid_str = str(sid)
                records.append(
                    {
                        "question_id": sid_str,
                        "question": q,
                        "ground_truth": ref,
                        "prediction": pred,
                    }
                )
                gts[sid_str] = [ref]
                hyps[sid_str] = [pred]

    metrics = compute_metrics(gts, hyps)
    logging.info("Evaluation metrics: %s", metrics)

    # Save qualitative results
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    logging.info("Saved detailed results to %s", os.path.abspath(args.output))

    # Print metrics for convenience
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()
