"""
Compute evaluation metrics (SPICE, BERTScore, ROUGE-L) from an existing BEV-QA
prediction JSON, without rerunning model generation.

Typical usage:
    python compute_bevqa_metrics.py \
        --input BEVDriver/LAVIS/lavis/out/bevqa/20251209074859/val_test_bevqa_epoch7.json
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List

from evaluate_bevqa import compute_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute BEV-QA metrics from saved predictions.")
    parser.add_argument(
        "--input",
        required=True,
        help="Path to JSON file containing a list of {id,pred,ref} records.",
    )
    parser.add_argument(
        "--output",
        help="Optional metrics output path (defaults to <input> with _metrics suffix).",
    )
    parser.add_argument(
        "--id-key",
        default="id",
        help="JSON field name for the sample id (fallbacks: question_id, sample_id).",
    )
    parser.add_argument(
        "--pred-key",
        default="pred",
        help="JSON field name for the prediction (fallbacks: prediction, answer).",
    )
    parser.add_argument(
        "--ref-key",
        default="ref",
        help="JSON field name for the reference text (fallbacks: ground_truth, gt, label).",
    )
    parser.add_argument(
        "--no-spice",
        action="store_true",
        help="Skip SPICE computation (avoids Java/pycocoevalcap dependency).",
    )
    parser.add_argument(
        "--pseudo-spice",
        action="store_true",
        help="Use lightweight pseudo-SPICE (token overlap F1) instead of full SPICE.",
    )
    parser.add_argument(
        "--no-bertscore",
        action="store_true",
        help="Skip BERTScore computation.",
    )
    parser.add_argument(
        "--no-rouge",
        action="store_true",
        help="Skip ROUGE-L computation.",
    )
    parser.add_argument(
        "--chrf",
        action="store_true",
        help="Compute chrF using sacrebleu (character-level F-score).",
    )
    parser.add_argument(
        "--bert-model",
        default="roberta-large",
        help="Model name for BERTScore (e.g., roberta-large, microsoft/deberta-xlarge-mnli).",
    )
    return parser.parse_args()


def _get_with_fallback(rec: Dict, primary: str, fallbacks: Iterable[str]):
    if primary in rec:
        return rec[primary]
    for key in fallbacks:
        if key in rec:
            return rec[key]
    raise KeyError(f"None of the keys {primary!r} or {list(fallbacks)} found in record: {rec}")


def load_predictions(
    path: Path,
    *,
    id_key: str,
    pred_key: str,
    ref_key: str,
) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list of records in {path}, got {type(data)}")

    records = []
    for rec in data:
        sample_id = _get_with_fallback(rec, id_key, ["question_id", "sample_id"])
        pred = _get_with_fallback(rec, pred_key, ["prediction", "answer"])
        ref = _get_with_fallback(rec, ref_key, ["ground_truth", "gt", "label"])
        records.append({"id": str(sample_id), "pred": str(pred), "ref": str(ref)})
    return records


def main():
    args = parse_args()

    if not logging.getLogger().hasHandlers():
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    input_path = Path(args.input)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_path = Path(args.output) if args.output else input_path.with_name(
        input_path.stem + "_metrics" + input_path.suffix
    )

    records = load_predictions(
        input_path,
        id_key=args.id_key,
        pred_key=args.pred_key,
        ref_key=args.ref_key,
    )

    gts: Dict[str, List[str]] = {r["id"]: [r["ref"]] for r in records}
    hyps: Dict[str, List[str]] = {r["id"]: [r["pred"]] for r in records}

    metrics = compute_metrics(
        gts,
        hyps,
        use_pseudo_spice=args.pseudo_spice,
        use_spice=not args.no_spice,
        use_bertscore=not args.no_bertscore,
        use_rouge=not args.no_rouge,
        use_chrf=args.chrf,
        bert_model=args.bert_model,
    )

    logging.info("Metrics: %s", metrics)
    print("Metrics:", metrics)

    metrics_obj = {"metrics": metrics, "num_examples": len(records)}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics_obj, f, ensure_ascii=False, indent=2)
    logging.info("Saved metrics to %s", output_path.resolve())
    print("Saved metrics to", output_path)


if __name__ == "__main__":
    main()
