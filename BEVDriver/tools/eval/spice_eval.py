import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger("spice_eval")

# Optional metric deps
try:
    from rouge_score import rouge_scorer
except ImportError:
    rouge_scorer = None  # type: ignore

try:
    from sacrebleu.metrics import CHRF
except ImportError:
    CHRF = None  # type: ignore

try:
    from bert_score import score as bert_score_fn
except ImportError:
    bert_score_fn = None  # type: ignore


def load_items(path: Path):
    with path.open("r") as f:
        return json.load(f)


def simple_f1(pred: str, ref: str) -> float:
    """
    Lightweight token-level F1 as a placeholder
    when true SPICE implementation is not available.
    """
    pred_tokens = pred.lower().split()
    ref_tokens = ref.lower().split()
    if not pred_tokens or not ref_tokens:
        return 0.0

    common = set(pred_tokens) & set(ref_tokens)
    if not common:
        return 0.0

    precision = len(common) / len(set(pred_tokens))
    recall = len(common) / len(set(ref_tokens))
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def evaluate_spice_like(pred_path: Path, ref_path: Path):
    preds = load_items(pred_path)
    refs = load_items(ref_path)

    # Expect list of {id, pred} and {id, ref} or a single dict keyed by id.
    if isinstance(preds, dict):
        pred_items = preds.items()
    else:
        pred_items = [(p["id"], p["pred"]) for p in preds]

    if isinstance(refs, dict):
        ref_map = refs
    else:
        ref_map = {r["id"]: r["ref"] for r in refs}

    scores = []
    for pid, ptxt in pred_items:
        rtxt = ref_map.get(pid, "")
        scores.append(simple_f1(ptxt, rtxt))

    if not scores:
        logging.warning("No overlapping ids between predictions and references.")
        return 0.0

    score = sum(scores) / len(scores)
    logging.info("Pseudo-SPICE (token F1) over %d items: %.4f", len(scores), score)
    return score


def evaluate_bevqa_file(path: Path) -> float:
    """
    Evaluate a single BEV-QA JSON file produced by BEVQADriveTask.
    Expected format:
      [
        {"id": "...", "pred": "...", "ref": "..."},
        ...
      ]
    """
    items = load_items(path)

    triples = []
    if isinstance(items, dict):
        # Allow dict-of-dicts fallback: {id: {"pred": ..., "ref": ...}, ...}
        for sid, rec in items.items():
            if not isinstance(rec, dict):
                continue
            triples.append((str(sid), rec.get("pred", ""), rec.get("ref", "")))
    else:
        for rec in items:
            if not isinstance(rec, dict):
                continue
            sid = str(rec.get("id", ""))
            pred = rec.get("pred", "")
            ref = rec.get("ref", "")
            triples.append((sid, pred, ref))

    if not triples:
        logging.warning("No valid records found in %s.", path)
        return 0.0

    scores = []
    for sid, pred, ref in triples:
        scores.append(simple_f1(pred, ref))

    score = sum(scores) / len(scores)
    logging.info(
        "Pseudo-SPICE (token F1) over %d BEV-QA items from %s: %.4f",
        len(scores),
        path,
        score,
    )
    return score


def compute_rouge_l(preds: List[str], refs: List[str]) -> Optional[float]:
    if rouge_scorer is None:
        _logger.warning("rouge_score is not installed; skipping ROUGE-L.")
        return None
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    vals = []
    for p, r in zip(preds, refs):
        score = scorer.score(r, p)["rougeL"].fmeasure  # type: ignore
        vals.append(score)
    return sum(vals) / len(vals) if vals else None


def compute_chrf(preds: List[str], refs: List[str]) -> Optional[float]:
    if CHRF is None:
        _logger.warning("sacrebleu is not installed; skipping chrF.")
        return None
    chrf = CHRF(word_order=0)  # chrF without word n-grams
    res = chrf.corpus_score(preds, [refs])
    return float(res.score) / 100.0 if res is not None else None


def compute_bertscore(
    preds: List[str],
    refs: List[str],
    model_type: str = "microsoft/deberta-base-mnli",
    batch_size: int = 32,
) -> Optional[float]:
    if bert_score_fn is None:
        _logger.warning("bert-score is not installed; skipping BERTScore.")
        return None
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, _, f1 = bert_score_fn(
        preds,
        refs,
        lang="en",
        model_type=model_type,
        batch_size=batch_size,
        device=device,
    )
    return float(f1.mean().item())


def evaluate_bevqa_with_metrics(
    path: Path,
    *,
    metrics: List[str],
    bert_model: str = "microsoft/deberta-base-mnli",
    bert_batch_size: int = 32,
) -> Dict[str, Optional[float]]:
    items = load_items(path)
    triples = []
    if isinstance(items, dict):
        for sid, rec in items.items():
            if not isinstance(rec, dict):
                continue
            triples.append((str(sid), rec.get("pred", ""), rec.get("ref", "")))
    else:
        for rec in items:
            if not isinstance(rec, dict):
                continue
            sid = str(rec.get("id", ""))
            pred = rec.get("pred", "")
            ref = rec.get("ref", "")
            triples.append((sid, pred, ref))

    preds = [p for _, p, _ in triples]
    refs = [r for _, _, r in triples]
    results: Dict[str, Optional[float]] = {}

    if "pseudo" in metrics:
        results["pseudo_f1"] = evaluate_bevqa_file(path)
    if "rougeL" in metrics:
        results["rougeL"] = compute_rouge_l(preds, refs)
        if results["rougeL"] is not None:
            _logger.info("ROUGE-L F1 over %d items: %.4f", len(preds), results["rougeL"])
    if "chrf" in metrics:
        results["chrf"] = compute_chrf(preds, refs)
        if results["chrf"] is not None:
            _logger.info("chrF over %d items: %.4f", len(preds), results["chrf"])
    if "bertscore" in metrics:
        results["bertscore_f1"] = compute_bertscore(
            preds, refs, model_type=bert_model, batch_size=bert_batch_size
        )
        if results["bertscore_f1"] is not None:
            _logger.info(
                "BERTScore F1 over %d items: %.4f", len(preds), results["bertscore_f1"]
            )
    return results


def main():
    parser = argparse.ArgumentParser(description="SPICE-style evaluation wrapper.")
    parser.add_argument(
        "--file",
        help="Single JSON file with fields id/pred/ref (e.g., val_bevqa_epoch*.json).",
    )
    parser.add_argument("--pred", help="Path to prediction JSON (legacy two-file mode).")
    parser.add_argument("--ref", help="Path to reference JSON (legacy two-file mode).")
    parser.add_argument(
        "--metrics",
        default="pseudo,rougeL,chrf,bertscore",
        help="Comma-separated metrics to compute (options: pseudo, rougeL, chrf, bertscore).",
    )
    parser.add_argument(
        "--bert-model",
        default="microsoft/deberta-base-mnli",
        help="Model name for BERTScore (if installed).",
    )
    parser.add_argument(
        "--bert-batch-size",
        type=int,
        default=32,
        help="Batch size for BERTScore (if installed).",
    )
    args = parser.parse_args()

    if args.file:
        path = Path(args.file)
        if not path.is_file():
            raise FileNotFoundError(path)
        metrics = [m.strip().lower() for m in args.metrics.split(",") if m.strip()]
        results = evaluate_bevqa_with_metrics(
            path,
            metrics=metrics,
            bert_model=args.bert_model,
            bert_batch_size=args.bert_batch_size,
        )
        _logger.info("Metrics summary: %s", json.dumps(results, indent=2))
    elif args.pred and args.ref:
        pred_path = Path(args.pred)
        ref_path = Path(args.ref)
        if not pred_path.is_file():
            raise FileNotFoundError(pred_path)
        if not ref_path.is_file():
            raise FileNotFoundError(ref_path)
        metrics = [m.strip().lower() for m in args.metrics.split(",") if m.strip()]
        # legacy two-file mode supports pseudo/rougeL/chrf/bertscore
        preds = load_items(pred_path)
        refs = load_items(ref_path)
        if isinstance(preds, dict):
            pred_list = list(preds.values())
        else:
            pred_list = list(preds)
        if isinstance(refs, dict):
            ref_list = list(refs.values())
        else:
            ref_list = list(refs)
        results = {}
        if "pseudo" in metrics:
            results["pseudo_f1"] = evaluate_spice_like(pred_path, ref_path)
        if "rougeL" in metrics:
            results["rougeL"] = compute_rouge_l(pred_list, ref_list)
        if "chrf" in metrics:
            results["chrf"] = compute_chrf(pred_list, ref_list)
        if "bertscore" in metrics:
            results["bertscore_f1"] = compute_bertscore(
                pred_list, ref_list, model_type=args.bert_model, batch_size=args.bert_batch_size
            )
        _logger.info("Metrics summary: %s", json.dumps(results, indent=2))
    else:
        parser.error("Provide either --file for BEV-QA JSON or both --pred and --ref.")


if __name__ == "__main__":
    main()
