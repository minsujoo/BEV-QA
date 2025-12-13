"""
Visualize BEV-QA predictions with multi-view images.

Usage:
  python BEVDriver/tools/visualize_bevqa.py \
    --result BEVDriver/LAVIS/lavis/out/bevqa/20251208050054/val_test_bevqa_epoch6.json \
    --sensor-root /workspace/Bench2Drive_Base \
    --out-dir /workspace/BEV-QA/vis_bevqa \
    --num 10

Expected JSON format (from bevqa_eval_best.py / BEVQADriveTask):
  [
    {"id": "<scenario>/<frame>", "pred": "...", "ref": "...", "question": "..."},
    ...
  ]

Images are loaded from:
  <sensor_root>/<scenario>/camera/{rgb_front,rgb_front_left,rgb_front_right}/<frame>.jpg
"""

import argparse
import json
import random
import textwrap
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
from PIL import Image


def load_json(path: Path) -> List[Dict]:
    with path.open("r") as f:
        return json.load(f)


def load_view(img_path: Path):
    if img_path.is_file():
        return Image.open(img_path).convert("RGB")
    return None


def wrap_text(text: str, width: int = 80) -> str:
    if not text:
        return ""
    wrapped_lines = []
    for line in text.splitlines():
        if not line.strip():
            wrapped_lines.append("")
            continue
        wrapped_lines.append(textwrap.fill(line, width=width))
    return "\n".join(wrapped_lines)


def extract_question_from_pred(pred: str):
    """
    If a prediction string embeds a 'Question:' header, split it into
    (question, remaining_prediction). Otherwise return (None, pred).
    """
    if not pred or "Question:" not in pred:
        return None, pred

    _, tail = pred.split("Question:", 1)
    tail = tail.strip()

    split_tokens = ["\nAnswer", "Answer:", "Answer in", "\nanswer", "answer:"]
    for tok in split_tokens:
        idx = tail.find(tok)
        if idx != -1:
            question = tail[:idx].strip()
            remaining = tail[idx:].lstrip()
            return question, remaining

    return tail, ""


def strip_prompt_from_pred(pred: str, question: str = None) -> str:
    """
    Remove repeated question/prompt lines from predictions so only the model answer remains.
    """
    if not pred:
        return pred

    qnorm = (question or "").strip().lower()
    cleaned = []
    for raw in pred.splitlines():
        line = raw.strip()
        lower = line.lower()
        if not line:
            continue
        if lower.startswith("question:"):
            continue
        if qnorm and lower.startswith(qnorm):
            continue
        if lower.startswith("answer"):
            keep = ""
            dot_idx = line.find(".")
            if dot_idx != -1:
                keep = line[dot_idx + 1 :].strip()
            if not keep:
                colon_idx = line.find(":")
                if colon_idx != -1:
                    keep = line[colon_idx + 1 :].strip()
            if not keep:
                continue
            cleaned.append(keep)
            continue
        cleaned.append(line)
    return " ".join(cleaned).strip()


def extract_first_human_turn(dialogue):
    """
    Extract the first human utterance from a chat-B2D style dialogue.
    """
    try:
        first_thread = dialogue[0]
        if isinstance(first_thread, list):
            for turn in first_thread:
                if isinstance(turn, dict) and turn.get("from") == "human":
                    return turn.get("value", "")
        if isinstance(first_thread, dict) and first_thread.get("from") == "human":
            return first_thread.get("value", "")
    except Exception:
        return ""
    return ""


def load_question_from_chatb2d(root: Path, sid: str) -> str:
    """
    Load the question from chat-B2D split folders (expects extracted files).
    """
    if not root or "/" not in sid:
        return ""
    scenario, frame = sid.split("/", 1)
    filename = f"{frame}.json"
    splits = ["val", "train"]

    for split in splits:
        file_path = root / split / scenario / filename
        if file_path.is_file():
            try:
                data = json.load(file_path.open())
                return extract_first_human_turn(data)
            except Exception:
                return ""

    return ""


def compose_figure(
    sample: Dict,
    sensor_root: Path,
    out_path: Path,
    question_lookup: Dict[str, str] = None,
    question_root: Path = None,
):
    sid = sample.get("id") or sample.get("question_id", "")
    question = (
        sample.get("question")
        or sample.get("prompt")
        or sample.get("instruction")
        or sample.get("input")
        or sample.get("query")
        or ""
    )
    if (not question) and question_lookup:
        question = question_lookup.get(sid, "")
    if (not question) and question_root:
        question = load_question_from_chatb2d(question_root, sid)
    pred = sample.get("pred") or sample.get("prediction", "")
    ref = sample.get("ref") or sample.get("ground_truth", "")

    if (not question) and pred:
        q_from_pred, cleaned_pred = extract_question_from_pred(pred)
        if q_from_pred:
            question = q_from_pred
            pred = cleaned_pred

    pred = strip_prompt_from_pred(pred, question)

    if "/" not in sid:
        return False
    scenario, frame = sid.split("/", 1)
    cam_dir = sensor_root / scenario / "camera"

    front = load_view(cam_dir / "rgb_front" / f"{frame}.jpg")
    left = load_view(cam_dir / "rgb_front_left" / f"{frame}.jpg")
    right = load_view(cam_dir / "rgb_front_right" / f"{frame}.jpg")

    if front is None:
        return False

    fig = plt.figure(figsize=(22, 14))
    gs = fig.add_gridspec(
        5,
        3,
        height_ratios=[1.0, 2.0, 2.0, 2.0, 0.1],
        width_ratios=[3.6, 4.2, 3.6],
    )

    ax_question = fig.add_subplot(gs[0, :])
    ax_question.axis("off")
    question_text = question.replace("\n", " ") if question else "(not provided)"
    ax_question.text(
        0.5,
        0.9,
        f"Question: {question_text}",
        fontsize=32,
        ha="center",
        va="top",
        transform=ax_question.transAxes,
    )

    ax_ref = fig.add_subplot(gs[1:4, 0])
    ax_ref.axis("off")
    ax_ref.text(
        0,
        1,
        "GT:\n" + wrap_text(ref, 32),
        fontsize=28,
        ha="left",
        va="top",
        transform=ax_ref.transAxes,
    )

    ax_left = fig.add_subplot(gs[1, 1])
    if left is not None:
        ax_left.imshow(left)
        ax_left.set_title("Front Left", fontsize=20, pad=14)
    ax_left.axis("off")

    ax_front = fig.add_subplot(gs[2, 1])
    ax_front.imshow(front)
    ax_front.set_title("Front", fontsize=20, pad=14)
    ax_front.axis("off")

    ax_right = fig.add_subplot(gs[3, 1])
    if right is not None:
        ax_right.imshow(right)
        ax_right.set_title("Front Right", fontsize=20, pad=14)
    ax_right.axis("off")

    ax_pred = fig.add_subplot(gs[1:4, 2])
    ax_pred.axis("off")
    ax_pred.text(
        0,
        1,
        "Prediction:\n" + wrap_text(pred, 32),
        fontsize=28,
        ha="left",
        va="top",
        transform=ax_pred.transAxes,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(top=0.92, bottom=0.01, left=0.03, right=0.97, hspace=0.25, wspace=0.06)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


def main():
    ap = argparse.ArgumentParser(description="Visualize BEV-QA predictions.")
    ap.add_argument("--result", required=True, help="Path to {id,pred,ref,question} JSON.")
    ap.add_argument("--sensor-root", required=True, help="Bench2Drive sensor root.")
    ap.add_argument("--out-dir", required=True, help="Output directory for figures.")
    ap.add_argument("--num", type=int, default=20, help="Number of samples to render.")
    ap.add_argument("--shuffle", action="store_true", help="Randomize sample order.")
    ap.add_argument(
        "--question-file",
        help="Optional JSON with {id,question,...} to fill missing question texts.",
    )
    ap.add_argument(
        "--question-root",
        default="/workspace/chat-B2D",
        help="Directory containing chat-B2D split folders (val/train) with per-frame JSON.",
    )
    args = ap.parse_args()

    result_path = Path(args.result)
    sensor_root = Path(args.sensor_root)
    out_dir = Path(args.out_dir)

    data = load_json(result_path)
    question_lookup = None
    if args.question_file:
        q_data = load_json(Path(args.question_file))
        question_lookup = {}
        for row in q_data:
            qid = row.get("id") or row.get("question_id")
            qtext = (
                row.get("question")
                or row.get("prompt")
                or row.get("instruction")
                or row.get("input")
                or row.get("query")
            )
            if qid and qtext:
                question_lookup[qid] = qtext
    question_root = Path(args.question_root) if args.question_root else None
    samples = data.copy()
    if args.shuffle:
        random.shuffle(samples)

    rendered = 0
    for sample in samples:
        sid = sample.get("id") or sample.get("question_id", "unknown")
        out_path = out_dir / f"{sid.replace('/', '_')}.png"
        ok = compose_figure(sample, sensor_root, out_path, question_lookup, question_root)
        if ok:
            rendered += 1
        if rendered >= args.num:
            break

    print(f"Rendered {rendered} samples to {out_dir}")


if __name__ == "__main__":
    main()
