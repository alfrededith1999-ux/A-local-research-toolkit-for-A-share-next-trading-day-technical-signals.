# Stock Limit-Up Research Toolkit

> A local research toolkit for A-share next-trading-day technical signals.  
> 本项目仅用于量化建模、股票技术交流和结果解释研究，不构成投资建议，也不应作为任何买卖决策的唯一依据。

## What This Project Does

This toolkit builds a local stock prediction workflow around structured market data:

- TuShare data collection GUI
- SQLite data storage
- XGBoost/RF tabular model
- LNN-style sequence model
- Transformer time-series branch
- Model registry for checkpoint continuation
- Three-run ensemble prediction
- Feedback backfill and stock/industry gradient calibration
- Result interpretation script
- LM Studio context export for explanation only

The language model is **not** part of the core scoring model. LM Studio can be used only to explain exported JSON results.

## Important Disclaimer

This repository is for technical research and educational discussion only.

- Not financial advice.
- Not a recommendation to buy, sell, hold, or short any security.
- Not a complete risk-control system.
- Prediction outputs can be wrong.
- Users are responsible for data legality, TuShare permissions, and their own decisions.

## Privacy And Token Safety

Do **not** commit your TuShare token.

This release does not contain a token. The collector reads the token from:

1. GUI input field, or
2. `TUSHARE_TOKEN` environment variable.

Recommended:

```bat
set "TUSHARE_TOKEN=your_tushare_token_here"
```

Never commit:

- `.env`
- local SQLite database files
- model checkpoints
- prediction outputs
- feedback buffer database
- TuShare token

The included `.gitignore` excludes these by default.

## Hardware Support

### Minimum Practical Hardware

CPU-only can run small experiments, but it is slow.

- CPU: 8 cores or better
- RAM: 32 GB minimum
- GPU: optional
- VRAM: 12 GB minimum if using CUDA
- Storage: SSD recommended

Suggested minimum GPU:

- RTX 3060 12GB
- RTX 4070 Ti SUPER 16GB
- RTX 4080/4080 SUPER 16GB
- RTX 3090 24GB
- RTX 4090/4090D 24GB

For low-VRAM GPUs, reduce:

- `max_train_rows`
- `lnn_batch_size`
- `transformer_batch_size`
- `transformer_d_model`
- `transformer_layers`
- `repeats`

### Recommended Hardware

- RAM: 64 GB or more
- GPU: RTX 4090/4090D 24GB or better
- VRAM: 24 GB or more
- SSD/NVMe storage

### Dual-GPU Enhanced Setup

Tested workflow target:

- RTX 5090D 32GB as primary
- RTX 4090D 24GB as secondary

Use:

```bat
set "CUDA_VISIBLE_DEVICES=1,0"
```

This maps the physical second GPU to runtime `cuda:0`, useful when the faster card is listed as device 1.

### Notes On NCCL / cuML

- NCCL warnings on Windows usually affect multi-GPU communication speed, not model correctness.
- cuML/cuDF are optional. On native Windows, RAPIDS cuML is usually not available; the toolkit falls back to XGBoost GPU or sklearn CPU.
- For RAPIDS, use WSL2/Linux and follow official RAPIDS installation instructions.

## Software Requirements

Recommended:

- Windows 10/11 or Linux
- Python 3.10-3.12 recommended
- NVIDIA Driver compatible with your PyTorch CUDA build
- CUDA-capable PyTorch if using GPU

Install common dependencies:

```bat
python -m pip install -U pip
python -m pip install -r requirements.txt
```

Install PyTorch from the official selector:

https://pytorch.org/get-started/locally/

## Project Files

```text
stock_limit_lnn_transformer_registry.py   Main GUI / model engine
run_prediction_three_times_registry.py    Headless three-run prediction launcher
tushare_gui_app_missing_only.py           TuShare collection GUI wrapper
interpret_results.py                      Latest result interpreter
diagnose_feedback_backfill.py             Feedback backfill diagnostic tool
repair_feedback_backfill.py               Feedback repair and gradient rebuild tool
config.example.json                       Example config with relative paths
requirements.txt                          Python dependencies
.gitignore                                Secret/output/model/data ignore rules
DISCLAIMER.md                             Research-only disclaimer
```

## Directory Layout

Recommended local layout:

```text
repo/
  data/
    tu_share_data.db
  ml_outputs/
  model_registry/
  ml_feedback_buffer.db
```

These runtime files are ignored by Git.

## Quick Start

### 1. Configure TuShare Token

Temporary environment variable:

```bat
set "TUSHARE_TOKEN=your_tushare_token_here"
```

Or enter it in the GUI. Do not commit it.

### 2. Collect Data

The GUI wrapper expects a compatible TuShare collection engine. If your engine is separate, place it next to the GUI or adapt the path in the GUI.

```bat
set "PYTHONUTF8=1" && set "PYTHONIOENCODING=utf-8" && python "tushare_gui_app_missing_only.py"
```

### 3. Run Three Predictions

Single GPU or default GPU order:

```bat
set "PYTHONUTF8=1" && set "PYTHONIOENCODING=utf-8" && python "run_prediction_three_times_registry.py" --config "config.example.json" --repeats 3
```

Dual GPU with device re-ordering:

```bat
set "PYTHONUTF8=1" && set "PYTHONIOENCODING=utf-8" && set "CUDA_VISIBLE_DEVICES=1,0" && python "run_prediction_three_times_registry.py" --config "config.example.json" --repeats 3
```

### 4. Interpret Results

```bat
set "PYTHONUTF8=1" && set "PYTHONIOENCODING=utf-8" && python "interpret_results.py" --output-dir "ml_outputs" --buffer-db "ml_feedback_buffer.db" --top 18
```

### 5. Diagnose Feedback

```bat
set "PYTHONUTF8=1" && set "PYTHONIOENCODING=utf-8" && python "diagnose_feedback_backfill.py" --buffer-db "ml_feedback_buffer.db" --data-db "data/tu_share_data.db"
```

### 6. Repair Feedback Backfill

```bat
set "PYTHONUTF8=1" && set "PYTHONIOENCODING=utf-8" && python "repair_feedback_backfill.py" --buffer-db "ml_feedback_buffer.db" --data-db "data/tu_share_data.db"
```

If table schema is incompatible:

```bat
set "PYTHONUTF8=1" && set "PYTHONIOENCODING=utf-8" && python "repair_feedback_backfill.py" --buffer-db "ml_feedback_buffer.db" --data-db "data/tu_share_data.db" --force-rebuild
```

## Model Registry

The model registry stores checkpoint metadata and weights:

```text
model_registry/
  latest.json
  runs/
    20260710_.../
      lnn_y_limit.pt
      lnn_y_limit.json
      transformer_y_limit.pt
      transformer_y_limit.json
```

It records:

- model kind
- target
- feature hash
- architecture parameters
- training loss
- quality metric
- config snapshot

If feature columns or architecture are incompatible, old weights are not loaded.

## LM Studio Explanation Layer

After prediction, the engine can export:

```text
ml_outputs/lmstudio_context_*.json
```

Use it in LM Studio with a prompt such as:

```text
Read this JSON. You are only the explanation layer and must not change model scoring.
First provide the overall conclusion, then key names/codes, sector reasoning, risks,
and intraday confirmation conditions. State clearly that this is technical research only.
```
