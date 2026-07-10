# -*- coding: utf-8 -*-
"""
三次运行新版：model_registry + LNN续训 + Transformer分支 + 三模型融合。

使用：
    python run_prediction_three_times_registry.py --config dualgpu_market_sector_config_registry.json --repeats 3
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import threading
import time
from dataclasses import asdict
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
ENGINE_PATH = BASE_DIR / "stock_limit_lnn_transformer_registry.py"


def load_engine_module():
    spec = importlib.util.spec_from_file_location("stock_limit_lnn_transformer_registry", ENGINE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载预测脚本：{ENGINE_PATH}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_config(mod, path: str):
    with open(path, "r", encoding="utf-8-sig") as f:
        data = json.load(f)
    allowed = set(mod.AppConfig.__dataclass_fields__.keys())
    clean = {k: v for k, v in data.items() if k in allowed}
    return mod.AppConfig(**clean)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=r"config.example.json")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    mod = load_engine_module()
    cfg0 = load_config(mod, args.config)
    print("=" * 80)
    print("三次预测启动：model_registry + LNN + Transformer + XGBoost/RF 融合")
    print(f"config={args.config}")
    print(f"model_registry={getattr(cfg0, 'model_registry_dir', '')}")
    print("=" * 80)

    results = []
    for i in range(1, int(args.repeats) + 1):
        cfg = load_config(mod, args.config)
        if hasattr(cfg, "rf_random_state"):
            cfg.rf_random_state = int(cfg.rf_random_state) + i * 101
        if hasattr(cfg, "lnn_random_state"):
            cfg.lnn_random_state = int(cfg.lnn_random_state) + i * 101
        if hasattr(cfg, "headless_repeat_index"):
            cfg.headless_repeat_index = i

        print("\n" + "=" * 80)
        print(f"开始第 {i}/{args.repeats} 次预测，rf_seed={getattr(cfg, 'rf_random_state', '')}, lnn_seed={getattr(cfg, 'lnn_random_state', '')}")
        print("=" * 80)

        def log(msg: str) -> None:
            print(msg, flush=True)

        def progress(pct: int, msg: str) -> None:
            print(f"[{pct:3d}%] {msg}", flush=True)

        stop_event = threading.Event()
        engine = mod.PredictionEngine(cfg, log, progress, stop_event)
        t0 = time.time()
        result = engine.run()
        result["elapsed_minutes"] = round((time.time() - t0) / 60.0, 2)
        results.append(result)

    print("\n" + "=" * 80)
    print("三次预测完成")
    for r in results:
        print(f"- run_id={r.get('run_id')} all={r.get('csv_all')} lmstudio={r.get('lmstudio_context')}")
    print("=" * 80)


if __name__ == "__main__":
    main()
