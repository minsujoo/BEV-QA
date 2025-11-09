# Repository Guidelines

## Project Structure & Module Organization
- Core code lives under `BEVDriver/`.
- Training/eval (LLM and pipelines): `BEVDriver/LAVIS/` (Python package under `LAVIS/lavis`).
- Encoder and data pipeline: `BEVDriver/timm/` (e.g., `timm/models/bevdriver_encoder_train.py`).
- Simulation & evaluation: `BEVDriver/leaderboard/`, `BEVDriver/scenario_runner/`.
- Data tooling: `BEVDriver/tools/` (e.g., `tools/data_preprocessing/*`).
- Encoder scripts: `BEVDriver/BEV_encoder/` (e.g., `train.sh`).
- Dependencies: `BEVDriver/requirements.txt`; CARLA setup: `BEVDriver/setup_carla.sh`.

## Build, Test, and Development Commands
- Create env: `conda create -n bevdriver python=3.8 && conda activate bevdriver`.
- Install deps: `pip install -r BEVDriver/requirements.txt`; then `cd BEVDriver/LAVIS && pip install -r requirements.txt && cd -` (use bundled LAVIS/timm; do not `pip install` upstream packages).
- CARLA: `chmod +x BEVDriver/setup_carla.sh && BEVDriver/setup_carla.sh && pip install carla`.
- Data prep (examples): `python BEVDriver/tools/data_preprocessing/index_routes.py dataset`.
- Train encoder: `bash BEVDriver/BEV_encoder/train.sh`.
- Train full model: `cd BEVDriver/LAVIS && ./run.sh 1 lavis/projects/bevdriver/train_modular.yaml`.
- Evaluate: `./BEVDriver/leaderboard/scripts/run_evaluation.sh` (set `SCENARIOS`/`ROUTES` as needed).

## Coding Style & Naming Conventions
- Python 3.8, PEP 8, 4‑space indent. Snake_case for functions/vars, PascalCase for classes, UPPER_SNAKE_CASE for constants.
- Prefer `logging` over `print` (e.g., `logging.getLogger("train")`).
- Keep functions small and focused; document non‑obvious logic with concise docstrings.
- Avoid refactoring vendored code in `LAVIS/`, `timm/`, `leaderboard/`, `scenario_runner/` unless necessary; match the existing style in those folders and clearly annotate local changes.

## Testing Guidelines
- Framework: pytest (tests under `BEVDriver/LAVIS/tests/`).
- Naming: files `test_*.py`, classes `Test*`, functions `test_*`.
- Run fast tests: `cd BEVDriver/LAVIS && pytest -q` (use `-k pattern` to filter). Add unit tests for new utilities where feasible. Integration tests that hit CARLA should be opt‑in.

## Commit & Pull Request Guidelines
- Commits: short, imperative subject (≤72 chars), descriptive body when needed. Example: `data_collection: fix Town01 long‑route scripts`.
- PRs must include: purpose and scope, reproduction commands (e.g., train/eval invocations), screenshots/logs when relevant, and any path/config changes. Link issues and note breaking changes.

## Security & Configuration Tips
- Do not commit datasets, checkpoints, or secrets. Use `.gitignore` and external storage.
- Keep paths configurable; avoid hard‑coding user‑specific directories. Redact proxies/keys (the `LAVIS/run.sh` proxy lines are examples, not required).
- Use bundled `LAVIS`/`timm`—do not replace with pip versions without discussion.
