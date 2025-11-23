"""
Evaluate a BEV-QA model using a saved checkpoint.

This script:
  - Loads a LAVIS config (e.g., lavis/projects/bevqa/train.yaml)
  - Builds the BEV-QA task, datasets, and model
  - Loads a user-specified checkpoint into the model
  - Runs evaluation on the validation split using BEVQADriveTask
  - Saves {id, pred, ref} JSON for downstream metrics (e.g., SPICE)

Example usage:
    conda activate bevdriver
    cd BEVDriver/LAVIS
    python bevqa_eval_best.py \
        --cfg-path lavis/projects/bevqa/train.yaml \
        --ckpt-path lavis/out/bevqa/20251121051756/checkpoint_best.pth

By default, this script:
  - Uses the config's valid_splits as test_splits if test_splits is empty.
  - Forces run.skip_generate = False so that predictions are generated.
"""

import argparse
import logging
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn

import lavis.tasks as tasks
from lavis.common.config import Config
from lavis.common.dist_utils import get_rank, init_distributed_mode
from lavis.common.logger import setup_logger
from lavis.common.utils import now
from lavis.runners.runner_base import RunnerBase

# imports modules for registration (datasets, models, processors, tasks)
from lavis.datasets.builders import *  # noqa: F401,F403
from lavis.models import *  # noqa: F401,F403
from lavis.processors import *  # noqa: F401,F403
from lavis.tasks import *  # noqa: F401,F403


def parse_args():
    parser = argparse.ArgumentParser(description="BEV-QA evaluation with a checkpoint.")

    parser.add_argument(
        "--cfg-path",
        required=True,
        help="Path to configuration file (e.g., lavis/projects/bevqa/train.yaml).",
    )
    parser.add_argument(
        "--ckpt-path",
        required=True,
        help="Path to checkpoint file to evaluate (e.g., .../checkpoint_best.pth).",
    )
    parser.add_argument(
        "--options",
        nargs="+",
        help=(
            "Override some settings in the used config, the key-value pair "
            "in xxx=yyy format will be merged into config file."
        ),
    )

    return parser.parse_args()


def setup_seeds(config: Config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


def main():
    job_id = now()
    args = parse_args()

    cfg = Config(args)

    # If no explicit test_splits are provided, use valid_splits for evaluation.
    if not cfg.run_cfg.get("test_splits", []):
        cfg.run_cfg.test_splits = cfg.run_cfg.valid_splits

    # For evaluation we typically want generation enabled so that BEVQADriveTask
    # can dump {id, pred, ref} JSON. Allow command-line overrides to take effect
    # via --options, but if not explicitly set, force skip_generate = False.
    if cfg.run_cfg.get("skip_generate", None) is None:
        cfg.run_cfg.skip_generate = False
    else:
        # Even if present in the config, user might want to override from CLI:
        #   --options run.skip_generate=false
        # so we do not overwrite an explicit value here.
        pass

    init_distributed_mode(cfg.run_cfg)
    setup_seeds(cfg)

    setup_logger()
    cfg.pretty_print()

    task = tasks.setup_task(cfg)
    datasets = task.build_datasets(cfg)
    model = task.build_model(cfg)

    # Load checkpoint weights into the freshly built model.
    logging.getLogger(__name__).info("Loading checkpoint from %s", args.ckpt_path)
    checkpoint = torch.load(args.ckpt_path, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    if missing:
        logging.getLogger(__name__).warning(
            "Missing keys when loading checkpoint: %s", ", ".join(missing)
        )
    if unexpected:
        logging.getLogger(__name__).warning(
            "Unexpected keys when loading checkpoint: %s", ", ".join(unexpected)
        )

    runner = RunnerBase(cfg=cfg, job_id=job_id, task=task, model=model, datasets=datasets)

    # Use the epoch stored in checkpoint (if available) only for naming outputs.
    cur_epoch = checkpoint.get("epoch", "best")
    runner.evaluate(cur_epoch=cur_epoch, skip_reload=True)


if __name__ == "__main__":
    main()

