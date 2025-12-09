"""
Simple batch scheduler to evaluate multiple BEV-QA checkpoints sequentially per GPU.

It discovers all checkpoint_best.pth files under lavis/out/bevqa/*/ and runs
bevqa_eval_best.py on each, keeping at most one job per GPU. When a job on a GPU
finishes, the next pending job is started on that GPU.

Usage (example):
  python BEVDriver/tools/eval/batch_bevqa_eval.py --gpus 0,1,2,3

Options:
  --cfg   : Path to train/eval config (default: lavis/projects/bevqa/train.yaml)
  --gpus  : Comma-separated GPU ids to use (default: 0,1,2,3)
  --splits: Test splits to evaluate (default: ["val_test"])
"""

import argparse
import glob
import os
import subprocess
import time
from pathlib import Path
from typing import Dict, List


def parse_args():
    ap = argparse.ArgumentParser(description="Batch BEV-QA eval scheduler.")
    ap.add_argument(
        "--cfg",
        default="/workspace/BEV-QA/BEVDriver/LAVIS/lavis/projects/bevqa/train.yaml",
        help="Config path for bevqa_eval_best.py",
    )
    ap.add_argument(
        "--gpus",
        default="0,1,2,3",
        help="Comma-separated GPU ids to use (one job per GPU).",
    )
    ap.add_argument(
        "--splits",
        default='["val_test"]',
        help='Test splits to evaluate, e.g., ["val_test"] or ["val_test","test"]',
    )
    return ap.parse_args()


def discover_checkpoints() -> List[Path]:
    base = Path("/workspace/BEV-QA/BEVDriver/LAVIS/lavis/out/bevqa")
    return sorted(base.glob("20*/checkpoint_best.pth"))


def main():
    args = parse_args()
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    if not gpus:
        raise ValueError("No GPUs provided.")

    ckpts = discover_checkpoints()
    if not ckpts:
        print("No checkpoints found under lavis/out/bevqa/*/checkpoint_best.pth")
        return

    pending = list(ckpts)
    active: Dict[str, subprocess.Popen] = {}
    logs: Dict[str, Path] = {}

    print(f"Found {len(pending)} checkpoints. GPUs: {gpus}")

    while pending or active:
        # Launch on free GPUs
        for gpu in list(gpus):
            if gpu in active:
                continue
            if not pending:
                break
            ckpt = pending.pop(0)
            run_dir = ckpt.parent
            log_path = Path(f"/workspace/BEV-QA/nohup_eval_{run_dir.name}.log")
            cmd = [
                "python",
                "/workspace/BEV-QA/BEVDriver/LAVIS/bevqa_eval_best.py",
                "--cfg-path",
                args.cfg,
                "--ckpt-path",
                str(ckpt),
                "--options",
                "run.world_size=1",
                "run.distributed=false",
                "run.skip_generate=false",
                "run.valid_splits=[]",
                f'run.test_splits={args.splits}',
            ]
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu
            # Ensure minimal dataloader threads to reduce CUDA stream usage
            env.setdefault("OMP_NUM_THREADS", "1")
            log_f = log_path.open("w")
            print(f"[LAUNCH] GPU {gpu} -> {ckpt} | log: {log_path}")
            proc = subprocess.Popen(cmd, stdout=log_f, stderr=log_f, env=env)
            active[gpu] = proc
            logs[gpu] = log_path

        # Check for finished jobs
        finished = []
        for gpu, proc in active.items():
            ret = proc.poll()
            if ret is not None:
                finished.append((gpu, ret))
        for gpu, ret in finished:
            print(f"[DONE] GPU {gpu} exit={ret} | log: {logs[gpu]}")
            active.pop(gpu, None)
            logs.pop(gpu, None)

        if pending or active:
            time.sleep(2)

    print("All evaluations completed.")


if __name__ == "__main__":
    main()
