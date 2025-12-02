"""
Evaluate a trained BEV-QA model on a validation split and report SPICE (primary),
plus BERTScore and ROUGE-L by default, while also saving per-example predictions for
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
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm

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
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=192,
        help="Max tokens to generate per answer (increase to avoid truncation).",
    )
    parser.add_argument(
        "--bert-model",
        default="roberta-large",
        help="Model name for BERTScore (e.g., roberta-large, microsoft/deberta-xlarge-mnli).",
    )
    return parser.parse_args()


def is_dist_avail_and_initialized():
    return dist.is_available() and dist.is_initialized()


def is_main_process():
    return (not is_dist_avail_and_initialized()) or dist.get_rank() == 0


def build_dataloader(task, cfg: Config, split: str, batch_size: int, num_workers: int):
    datasets = task.build_datasets(cfg)
    # Use the first declared dataset (e.g., bench2drive_chatb2d)
    dataset_name = next(iter(datasets.keys()))
    if split not in datasets[dataset_name]:
        raise ValueError(f"Split '{split}' not found in dataset '{dataset_name}'. "
                         f"Available splits: {list(datasets[dataset_name].keys())}")
    dataset = datasets[dataset_name][split]

    collate_fn = getattr(dataset, "collater", None)
    sampler = None
    if is_dist_avail_and_initialized():
        sampler = DistributedSampler(dataset, shuffle=False)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False if sampler is None else False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        sampler=sampler,
    )
    return dataloader


def move_to_device(samples: Dict, device: torch.device) -> Dict:
    # Use existing helper to move tensors; keeps strings/lists intact.
    return prepare_sample(samples, cuda_enabled=device.type == "cuda")


def compute_metrics(
    gts: Dict,
    hyps: Dict,
    *,
    use_spice: bool = True,
    use_bertscore: bool = True,
    use_rouge: bool = True,
    bert_model: str = None,
) -> Dict[str, float]:
    metrics = {}

    if use_spice:
        from pycocoevalcap.spice.spice import Spice

        spice_scorer = Spice()
        spice_score, _ = spice_scorer.compute_score(gts, hyps)
        metrics["SPICE"] = spice_score

    if use_bertscore:
        try:
            from bert_score import score as bert_score
        except ImportError:
            logging.warning("bert_score is not installed; skipping BERTScore.")
        else:
            if bert_model is None:
                bert_model = "roberta-large"
            # Flatten refs/hyps using consistent key order.
            keys = sorted(gts.keys())
            refs = [gts[k][0] for k in keys]
            hyps_list = [hyps[k][0] for k in keys]
            P, R, F1 = bert_score(
                hyps_list,
                refs,
                model_type=bert_model if bert_model else None,
                verbose=False,
            )
            metrics["BERTScore_P"] = float(P.mean())
            metrics["BERTScore_R"] = float(R.mean())
            metrics["BERTScore_F1"] = float(F1.mean())

    if use_rouge:
        try:
            from rouge_score import rouge_scorer
        except ImportError:
            logging.warning("rouge_score is not installed; skipping ROUGE-L.")
        else:
            scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
            keys = sorted(gts.keys())
            refs = [gts[k][0] for k in keys]
            hyps_list = [hyps[k][0] for k in keys]
            scores = [scorer.score(ref, hyp)["rougeL"].fmeasure for ref, hyp in zip(refs, hyps_list)]
            if scores:
                metrics["ROUGE-L"] = float(sum(scores) / len(scores))

    return metrics


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # Distributed setup (torchrun)
    if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        args.device = "cuda"
        if is_main_process():
            logging.info("Running distributed evaluation with world_size=%d", dist.get_world_size())
    else:
        local_rank = None

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

    progress = tqdm(
        total=len(dataloader),
        disable=not is_main_process(),
        desc="Evaluating",
    )

    with torch.no_grad():
        for samples in dataloader:
            samples = move_to_device(samples, device)

            preds = model.generate(samples, max_new_tokens=args.max_new_tokens)
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

            progress.update(1)

    progress.close()

    # Gather results from all ranks.
    if is_dist_avail_and_initialized():
        gather_list = None
        obj = {"records": records, "gts": gts, "hyps": hyps}
        if is_main_process():
            gather_list = [None for _ in range(dist.get_world_size())]
        dist.gather_object(obj, gather_list, dst=0)

        if is_main_process():
            merged_records = []
            merged_gts = {}
            merged_hyps = {}
            for item in gather_list:
                if not item:
                    continue
                merged_records.extend(item["records"])
                merged_gts.update(item["gts"])
                merged_hyps.update(item["hyps"])
            records, gts, hyps = merged_records, merged_gts, merged_hyps
    # If not distributed, records/gts/hyps are already local.

    if is_main_process():
        metrics = compute_metrics(
            gts,
            hyps,
            use_spice=True,
            use_bertscore=True,
            use_rouge=True,
            bert_model=args.bert_model,
        )
        logging.info("Evaluation metrics: %s", metrics)

        # Save qualitative results (per-example predictions).
        output_path = Path(args.output)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        logging.info("Saved detailed results to %s", os.path.abspath(str(output_path)))

        # Save aggregate metrics alongside, in a separate JSON file.
        metrics_path = output_path.with_name(output_path.stem + "_metrics" + output_path.suffix)
        metrics_obj = {
            "metrics": metrics,
            "num_examples": len(records),
        }
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics_obj, f, ensure_ascii=False, indent=2)
        logging.info("Saved aggregate metrics to %s", os.path.abspath(str(metrics_path)))

        # Print metrics for convenience
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")

    if is_dist_avail_and_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
