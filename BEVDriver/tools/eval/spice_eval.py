import argparse
import json
import logging
from pathlib import Path


logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger("spice_eval")


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


def main():
    parser = argparse.ArgumentParser(description="SPICE-style evaluation wrapper.")
    parser.add_argument(
        "--file",
        help="Single JSON file with fields id/pred/ref (e.g., val_bevqa_epoch*.json).",
    )
    parser.add_argument("--pred", help="Path to prediction JSON (legacy two-file mode).")
    parser.add_argument("--ref", help="Path to reference JSON (legacy two-file mode).")
    args = parser.parse_args()

    if args.file:
        path = Path(args.file)
        if not path.is_file():
            raise FileNotFoundError(path)
        evaluate_bevqa_file(path)
    elif args.pred and args.ref:
        pred_path = Path(args.pred)
        ref_path = Path(args.ref)
        if not pred_path.is_file():
            raise FileNotFoundError(pred_path)
        if not ref_path.is_file():
            raise FileNotFoundError(ref_path)
        evaluate_spice_like(pred_path, ref_path)
    else:
        parser.error("Provide either --file for BEV-QA JSON or both --pred and --ref.")


if __name__ == "__main__":
    main()
