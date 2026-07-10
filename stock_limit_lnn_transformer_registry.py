# -*- coding: utf-8 -*-
"""
股票次日涨停/大涨预测 GUI
------------------------------------------------------------
功能：
1) 手动输入 SQLite 数据库路径，例如 F:\\股票\\tu_share_data.db
2) 自动读取 TuShare 采集库中的 daily / daily_basic / moneyflow / stock_basic 等表
3) 生成时序特征，训练随机森林/ExtraTrees + 液态神经网络(LTC/LNN近似实现)
4) 支持自由选择路径：RF->LNN、LNN->RF、RF only、LNN only、Ensemble
5) 支持勾选 4090D / 5090D / 同时使用；随机森林使用 CPU 多线程，LNN 使用 PyTorch GPU
6) 输出候选股：涨停概率、大涨概率、综合分；自动拆分低价/高价板块
7) 新增量价形态：低位缩量上涨、高位放量下跌，既进入模型特征，也可修正最终分数
8) 结果写回同一数据库，并导出 CSV / XLSX
9) 预留外部运算脚本接口：可把当前 DB 与输出目录传给后续脚本

注意：
- 本程序不是投资建议，只是量化建模工具。
- 液态神经网络为工程可运行版的 Liquid Time-Constant 风格循环单元，避免依赖 ncps 等额外库。
- 如果你的数据库非常大，建议先用较短训练起始日期和较小 max_train_rows 测试。
"""

from __future__ import annotations

import os
import sys
import json
import math
import time
import sqlite3
import random
import threading
import traceback
import subprocess
import pickle
import hashlib
from dataclasses import dataclass, asdict, replace
from pathlib import Path
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional, Tuple

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score, average_precision_score
except Exception as e:
    raise RuntimeError("请先安装 scikit-learn：python -m pip install -U scikit-learn") from e


# -----------------------------
# 配置结构
# -----------------------------

@dataclass
class AppConfig:
    db_path: str
    output_dir: str
    start_date: str = "20200101"
    exclude_bj: bool = True
    exclude_st: bool = True
    min_amount: float = 50000.0      # daily.amount 通常单位为千元，不同库可能有差异；这里只作为流动性粗筛
    min_close: float = 1.0
    max_close: float = 300.0
    low_price_cutoff: float = 20.0
    limit_threshold: float = 9.5
    bigrise_threshold: float = 5.0
    final_topn_each_band: int = 100
    first_stage_topn: int = 800
    max_train_rows: int = 250000
    validation_days_ratio: float = 0.18
    feature_history_days: int = 900   # 日常预测默认只读取近约900个交易日，减少大库读取和rolling计算

    pathway: str = "RF_TO_LNN"       # RF_TO_LNN / LNN_TO_RF / RF_ONLY / LNN_ONLY / ENSEMBLE
    tree_model: str = "RandomForest" # RandomForest / ExtraTrees
    tree_backend: str = "auto"        # auto / cuml / xgboost_gpu / sklearn
    rf_n_estimators: int = 600
    rf_max_depth: int = 10
    rf_min_samples_leaf: int = 20
    rf_n_jobs: int = -1
    rf_random_state: int = 2026
    rf_class_weight: bool = True

    lnn_seq_len: int = 20
    lnn_hidden_size: int = 96
    lnn_epochs: int = 8
    lnn_batch_size: int = 1024
    lnn_lr: float = 1e-3
    lnn_dropout: float = 0.10
    lnn_random_state: int = 2026

    use_4090d: bool = True
    use_5090d: bool = True
    allow_cpu_fallback: bool = True
    preferred_primary_gpu: int = 0      # 5090D 32G，作为主卡
    preferred_secondary_gpu: int = 1    # 4090D 24G，作为协同卡
    use_dual_gpu_parallel: bool = True

    weight_limit: float = 0.65
    weight_bigrise: float = 0.35

    # 量价形态修正：低位缩量上涨加分，高位放量下跌扣分
    use_vp_pattern: bool = True
    vp_low_pos_cutoff: float = 0.35       # 低位阈值：20/60日区间位置 <= 0.35
    vp_high_pos_cutoff: float = 0.70      # 高位阈值：20/60日区间位置 >= 0.70
    vp_shrink_cutoff: float = 0.80        # 缩量阈值：当前量/均量 <= 0.80
    vp_expand_cutoff: float = 1.50        # 放量阈值：当前量/均量 >= 1.50
    low_volume_up_bonus: float = 0.06     # 低位缩量上涨对 final_score 的奖励系数
    high_volume_down_penalty: float = 0.10 # 高位放量下跌对 final_score 的惩罚系数

    # 马尔可夫-贝叶斯式三日回测自动选参
    # 核心思想：今天的形态并不直接决定明天涨停，而是改变“明日涨停/大涨”的后验概率。
    # 因此先用最近3个已知交易日做滚动预测验证，给参数组合打分，再用得分最高的参数做最终预测。
    auto_param_search: bool = True
    param_search_days: int = 3
    param_search_topn: int = 80
    param_search_candidates: int = 4
    param_search_max_train_rows: int = 60000
    param_search_max_trees: int = 220
    param_search_min_history_days: int = 120

    # 分池输出与防过拟合控制
    use_pool_output: bool = True
    pool_topn_each: int = 30
    max_auto_weight_limit: float = 0.70
    startup_min_pct: float = 2.0
    latent_min_pct: float = 0.0
    latent_max_pct: float = 5.0
    latent_low_signal_min: float = 0.05
    risk_high_signal_min: float = 0.25

    # 大盘/板块与对抗验证
    use_market_sector_branch: bool = True
    use_adversarial_validation: bool = True
    drift_penalty_strength: float = 0.035
    sector_topn: int = 60

    # 权重注册表 / LNN + Transformer 延续训练 / LM Studio 解读导出
    use_model_registry: bool = True
    model_registry_dir: str = r"model_registry"
    continue_lnn_from_latest: bool = True
    continue_transformer_from_latest: bool = True
    registry_min_metric_keep_ratio: float = 0.97

    use_transformer_branch: bool = True
    transformer_seq_len: int = 32
    transformer_d_model: int = 128
    transformer_heads: int = 4
    transformer_layers: int = 2
    transformer_dropout: float = 0.12
    transformer_epochs: int = 6
    transformer_batch_size: int = 768
    transformer_lr: float = 8e-4

    use_registry_fusion: bool = True
    fusion_weight_tree: float = 0.40
    fusion_weight_lnn: float = 0.30
    fusion_weight_transformer: float = 0.30

    export_lmstudio_context: bool = True
    lmstudio_context_topn: int = 40


# -----------------------------
# SQLite 工具
# -----------------------------

def connect_db(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, timeout=120)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    conn.execute("PRAGMA cache_size=-300000;")
    conn.execute("PRAGMA busy_timeout=120000;")
    try:
        conn.execute("PRAGMA mmap_size=1073741824;")
    except Exception:
        pass
    return conn


def ensure_fast_read_indexes(conn: sqlite3.Connection, log: Optional[Callable[[str], None]] = None) -> None:
    """给日常读取和回写常用字段补轻量索引；已存在时不会重复创建。"""
    specs = [
        ("daily", "idx_daily_ts_date_fast", ["ts_code", "trade_date"]),
        ("daily", "idx_daily_date_fast", ["trade_date"]),
        ("daily_basic", "idx_daily_basic_ts_date_fast", ["ts_code", "trade_date"]),
        ("moneyflow", "idx_moneyflow_ts_date_fast", ["ts_code", "trade_date"]),
        ("stock_basic", "idx_stock_basic_ts_fast", ["ts_code"]),
        ("trade_calendar", "idx_trade_calendar_cal_fast", ["cal_date"]),
    ]
    made = 0
    for table, index_name, cols in specs:
        try:
            existing_cols = set(get_table_cols(conn, table))
            if not existing_cols or any(c not in existing_cols for c in cols):
                continue
            col_sql = ", ".join(_quote_ident(c) for c in cols)
            conn.execute(f'CREATE INDEX IF NOT EXISTS {_quote_ident(index_name)} ON {_quote_ident(table)} ({col_sql});')
            made += 1
        except Exception:
            pass
    try:
        conn.commit()
    except Exception:
        pass
    if log and made:
        log("已检查/补齐读取加速索引，后续读库和合并会更快。")


def append_df_fast(df: pd.DataFrame, table: str, conn: sqlite3.Connection) -> None:
    if df is None or df.empty:
        return
    df.to_sql(table, conn, if_exists="append", index=False, chunksize=5000, method=None)


def table_exists(conn: sqlite3.Connection, table: str) -> bool:
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=? LIMIT 1", (table,))
    return cur.fetchone() is not None


def get_table_cols(conn: sqlite3.Connection, table: str) -> List[str]:
    if not table_exists(conn, table):
        return []
    cur = conn.cursor()
    cur.execute(f'PRAGMA table_info("{table}")')
    return [str(r[1]) for r in cur.fetchall()]




def _quote_ident(name: str) -> str:
    """安全包装 SQLite 标识符，避免列名中出现特殊字符时 ALTER 失败。"""
    return '"' + str(name).replace('"', '""') + '"'


def _sqlite_type_from_series(series: pd.Series) -> str:
    """根据 pandas dtype 给 SQLite 新增列选择一个宽松类型。"""
    if pd.api.types.is_integer_dtype(series):
        return "INTEGER"
    if pd.api.types.is_float_dtype(series) or pd.api.types.is_bool_dtype(series):
        return "REAL"
    return "TEXT"


def ensure_table_accepts_dataframe(
    conn: sqlite3.Connection,
    table: str,
    df: pd.DataFrame,
    log: Optional[Callable[[str], None]] = None,
):
    """
    pandas.to_sql(if_exists='append') 要求 DataFrame 中的所有列都已存在于旧表。
    旧版本程序已经创建过 ml_prediction_candidates 时，新版本新增的量价形态列
    会触发 sqlite3.OperationalError: no column named xxx。

    这个函数会在写入前自动 ALTER TABLE ADD COLUMN，实现旧库无损迁移。
    如果表不存在，不处理，让 pandas.to_sql 首次创建表。
    """
    if df is None or df.empty or not table_exists(conn, table):
        return

    existing = set(get_table_cols(conn, table))
    missing = [c for c in df.columns if c not in existing]
    if not missing:
        return

    cur = conn.cursor()
    for col in missing:
        col_type = _sqlite_type_from_series(df[col])
        cur.execute(f'ALTER TABLE {_quote_ident(table)} ADD COLUMN {_quote_ident(col)} {col_type};')
    conn.commit()
    if log:
        log(f"数据库表 {table} 自动补齐新增列 {len(missing)} 个：" + ", ".join(missing[:12]) + (" ..." if len(missing) > 12 else ""))


# -----------------------------
# model_registry 权重注册表
# -----------------------------

class ModelRegistry:
    def __init__(self, root_dir: str, log: Callable[[str], None]):
        self.root = Path(root_dir)
        self.log = log
        self.latest_path = self.root / "latest.json"
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "runs").mkdir(parents=True, exist_ok=True)

    @staticmethod
    def feature_hash(feature_cols: List[str]) -> str:
        raw = json.dumps(list(feature_cols), ensure_ascii=False, sort_keys=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def key(model_kind: str, target: str) -> str:
        return f"{model_kind}_{target}"

    def _load_latest(self) -> Dict[str, dict]:
        if not self.latest_path.exists():
            return {}
        try:
            with self.latest_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _save_latest(self, latest: Dict[str, dict]) -> None:
        tmp = self.latest_path.with_suffix(".json.tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(latest, f, ensure_ascii=False, indent=2)
        tmp.replace(self.latest_path)

    def compatible_latest(self, model_kind: str, target: str, feature_cols: List[str], expected: Dict[str, object]) -> Optional[dict]:
        entry = self._load_latest().get(self.key(model_kind, target))
        if not entry:
            return None
        meta_path = Path(entry.get("meta_path", ""))
        weight_path = Path(entry.get("weight_path", ""))
        if not meta_path.exists() or not weight_path.exists():
            return None
        try:
            with meta_path.open("r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            return None
        if meta.get("feature_hash") != self.feature_hash(feature_cols):
            return None
        for k, v in expected.items():
            if meta.get(k) != v:
                return None
        out = dict(entry)
        out["meta"] = meta
        return out

    def save_torch_model(
        self,
        model,
        model_kind: str,
        target: str,
        cfg: AppConfig,
        feature_cols: List[str],
        arch_meta: Dict[str, object],
        metrics: Dict[str, float],
        extra_meta: Optional[Dict[str, object]] = None,
    ) -> Dict[str, object]:
        torch = import_torch()[0]
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        key = self.key(model_kind, target)
        run_dir = self.root / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        weight_path = run_dir / f"{key}.pt"
        meta_path = run_dir / f"{key}.json"
        core_model = model.module if hasattr(model, "module") else model
        torch.save(core_model.state_dict(), str(weight_path))

        meta = {
            "key": key,
            "model_kind": model_kind,
            "target": target,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "feature_hash": self.feature_hash(feature_cols),
            "feature_count": len(feature_cols),
            "feature_columns": list(feature_cols),
            "metrics": metrics,
            "quality_metric": float(metrics.get("quality_metric", 0.0)),
            "config_snapshot": asdict(cfg),
        }
        meta.update(arch_meta)
        if extra_meta:
            meta.update(extra_meta)
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        latest = self._load_latest()
        prev = latest.get(key)
        prev_metric = float((prev or {}).get("quality_metric", -1.0))
        new_metric = float(meta["quality_metric"])
        ratio = float(getattr(cfg, "registry_min_metric_keep_ratio", 0.97) or 0.97)
        accepted = prev is None or new_metric >= prev_metric * ratio
        if accepted:
            latest[key] = {
                "created_at": meta["created_at"],
                "weight_path": str(weight_path),
                "meta_path": str(meta_path),
                "quality_metric": new_metric,
                "metrics": metrics,
            }
            self._save_latest(latest)
            self.log(f"model_registry 已更新 latest：{key}, quality={new_metric:.6f}")
        else:
            self.log(f"model_registry 未更新 latest：{key}, 新质量 {new_metric:.6f} < 旧质量 {prev_metric:.6f} * {ratio:.3f}")
        return {"accepted_latest": accepted, "weight_path": str(weight_path), "meta_path": str(meta_path), "quality_metric": new_metric}


def read_existing_cols(
    conn: sqlite3.Connection,
    table: str,
    desired_cols: List[str],
    start_date: Optional[str] = None,
    date_col: str = "trade_date",
    log: Optional[Callable[[str], None]] = None,
) -> pd.DataFrame:
    if not table_exists(conn, table):
        if log:
            log(f"表 {table} 不存在，跳过。")
        return pd.DataFrame()
    cols = get_table_cols(conn, table)
    use_cols = [c for c in desired_cols if c in cols]
    if not use_cols:
        if log:
            log(f"表 {table} 没有可用字段，跳过。")
        return pd.DataFrame()
    if "ts_code" in cols and "ts_code" not in use_cols:
        use_cols.insert(0, "ts_code")
    if date_col in cols and date_col not in use_cols:
        use_cols.insert(1 if "ts_code" in use_cols else 0, date_col)
    col_sql = ", ".join([f'"{c}"' for c in use_cols])
    order_sql = ""
    if "ts_code" in use_cols and date_col in use_cols:
        order_sql = f' ORDER BY "ts_code", "{date_col}"'
    elif date_col in use_cols:
        order_sql = f' ORDER BY "{date_col}"'
    if start_date and date_col in cols:
        sql = f'SELECT {col_sql} FROM "{table}" WHERE "{date_col}">=?{order_sql}'
        df = pd.read_sql(sql, conn, params=(start_date,))
    else:
        sql = f'SELECT {col_sql} FROM "{table}"{order_sql}'
        df = pd.read_sql(sql, conn)
    if log:
        log(f"读取 {table}: {len(df):,} 行，字段 {len(use_cols)} 个。")
    return df


def safe_to_numeric(df: pd.DataFrame, exclude: Optional[List[str]] = None) -> pd.DataFrame:
    exclude = set(exclude or [])
    out = df.copy()
    for c in out.columns:
        if c in exclude:
            continue
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def fast_group_rolling(g, column: str, window: int, min_periods: int, op: str) -> pd.Series:
    rolled = g[column].rolling(window, min_periods=min_periods)
    if op == "mean":
        out = rolled.mean()
    elif op == "std":
        out = rolled.std()
    elif op == "max":
        out = rolled.max()
    elif op == "min":
        out = rolled.min()
    else:
        raise ValueError(f"Unsupported rolling op: {op}")
    return out.reset_index(level=0, drop=True)


# -----------------------------
# 数据与特征工程
# -----------------------------

class FeatureBuilder:
    def __init__(self, cfg: AppConfig, log: Callable[[str], None], progress: Callable[[int, str], None], stop_event: threading.Event):
        self.cfg = cfg
        self.log = log
        self.progress = progress
        self.stop_event = stop_event
        self.param_search_scores_df: Optional[pd.DataFrame] = None
        self.param_search_predictions_df: Optional[pd.DataFrame] = None
        self.selected_param_summary: Dict[str, object] = {}

    def _check_stop(self):
        if self.stop_event.is_set():
            raise RuntimeError("用户已停止任务。")

    def _effective_start_date(self, conn: sqlite3.Connection) -> str:
        start_date = str(self.cfg.start_date or "").strip()
        lookback = int(getattr(self.cfg, "feature_history_days", 0) or 0)
        if lookback <= 0 or not table_exists(conn, "daily") or "trade_date" not in get_table_cols(conn, "daily"):
            return start_date
        try:
            latest_df = pd.read_sql('SELECT MAX(trade_date) AS d FROM "daily"', conn)
            latest = str(latest_df.loc[0, "d"]) if not latest_df.empty and latest_df.loc[0, "d"] is not None else ""
            latest_dt = datetime.strptime(latest[:8], "%Y%m%d")
            cutoff = (latest_dt - timedelta(days=int(lookback * 1.65) + 120)).strftime("%Y%m%d")
            if start_date:
                cutoff = max(start_date, cutoff)
            self.log(f"加速读取：本次从 {cutoff} 开始读取历史数据，保留约 {lookback} 个交易日用于建模。")
            return cutoff
        except Exception as e:
            self.log(f"计算加速读取起点失败，继续使用窗口起始日期 {start_date}：{e}")
            return start_date

    def load_and_merge(self) -> pd.DataFrame:
        self.progress(5, "连接数据库")
        conn = connect_db(self.cfg.db_path)
        try:
            ensure_fast_read_indexes(conn, self.log)
            effective_start_date = self._effective_start_date(conn)
            self.progress(8, "读取日线数据")
            daily_cols = [
                "ts_code", "trade_date", "open", "high", "low", "close", "pre_close",
                "change", "pct_chg", "vol", "amount"
            ]
            daily = read_existing_cols(conn, "daily", daily_cols, effective_start_date, "trade_date", self.log)
            if daily.empty:
                raise RuntimeError("daily 表为空或不存在，无法建模。请先完成 TuShare 日线采集。")
            daily = daily.drop_duplicates(["ts_code", "trade_date"]).copy()
            daily = safe_to_numeric(daily, exclude=["ts_code", "trade_date"])
            daily["trade_date"] = daily["trade_date"].astype(str)

            self.progress(12, "读取基础行情扩展数据")
            basic_cols = [
                "ts_code", "trade_date", "turnover_rate", "turnover_rate_f", "volume_ratio",
                "pe", "pe_ttm", "pb", "ps", "ps_ttm", "dv_ratio", "dv_ttm",
                "total_share", "float_share", "free_share", "total_mv", "circ_mv"
            ]
            daily_basic = read_existing_cols(conn, "daily_basic", basic_cols, effective_start_date, "trade_date", self.log)
            if not daily_basic.empty:
                daily_basic = daily_basic.drop_duplicates(["ts_code", "trade_date"]).copy()
                daily_basic = safe_to_numeric(daily_basic, exclude=["ts_code", "trade_date"])
                daily = daily.merge(daily_basic, on=["ts_code", "trade_date"], how="left")

            self.progress(16, "读取资金流数据")
            mf_cols = [
                "ts_code", "trade_date", "buy_sm_vol", "buy_sm_amount", "sell_sm_vol", "sell_sm_amount",
                "buy_md_vol", "buy_md_amount", "sell_md_vol", "sell_md_amount",
                "buy_lg_vol", "buy_lg_amount", "sell_lg_vol", "sell_lg_amount",
                "buy_elg_vol", "buy_elg_amount", "sell_elg_vol", "sell_elg_amount",
                "net_mf_vol", "net_mf_amount"
            ]
            mf = read_existing_cols(conn, "moneyflow", mf_cols, effective_start_date, "trade_date", self.log)
            if not mf.empty:
                mf = mf.drop_duplicates(["ts_code", "trade_date"]).copy()
                mf = safe_to_numeric(mf, exclude=["ts_code", "trade_date"])
                daily = daily.merge(mf, on=["ts_code", "trade_date"], how="left")

            self.progress(20, "读取股票名称和行业")
            stock_cols = ["ts_code", "name", "area", "industry", "market", "list_date"]
            stock_basic = read_existing_cols(conn, "stock_basic", stock_cols, None, "trade_date", self.log)
            if not stock_basic.empty and "ts_code" in stock_basic.columns:
                stock_basic = stock_basic.drop_duplicates(["ts_code"]).copy()
                daily = daily.merge(stock_basic, on="ts_code", how="left")
            else:
                daily["name"] = ""
                daily["industry"] = ""
                daily["market"] = ""
                daily["list_date"] = np.nan
        finally:
            conn.close()

        self._check_stop()
        return daily

    def build_features(self, raw: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], str, str]:
        self.progress(24, "清洗与过滤股票池")
        df = raw.copy()
        df["ts_code"] = df["ts_code"].astype(str)
        df["trade_date"] = df["trade_date"].astype(str)

        if self.cfg.exclude_bj:
            df = df[~df["ts_code"].str.endswith(".BJ", na=False)].copy()
        if self.cfg.exclude_st and "name" in df.columns:
            df["name"] = df["name"].fillna("").astype(str)
            df = df[~df["name"].str.contains("ST|退", case=False, regex=True, na=False)].copy()

        for col in ["open", "high", "low", "close", "pre_close", "pct_chg", "vol", "amount"]:
            if col not in df.columns:
                df[col] = np.nan
            df[col] = pd.to_numeric(df[col], errors="coerce")

        if df["pct_chg"].isna().all() and not df["close"].isna().all():
            df["pct_chg"] = df.groupby("ts_code")["close"].pct_change() * 100.0

        df = df[(df["close"] >= self.cfg.min_close) & (df["close"] <= self.cfg.max_close)].copy()
        if "amount" in df.columns and self.cfg.min_amount > 0:
            df = df[(df["amount"].fillna(0) >= self.cfg.min_amount)].copy()

        if df.empty:
            raise RuntimeError("过滤后数据为空。请降低 min_amount/min_close/max_close 或检查 daily 表。")

        df = df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
        latest_trade_date = str(df["trade_date"].max())
        next_trade_date = self._guess_next_trade_date(latest_trade_date)
        self.log(f"建模最新交易日: {latest_trade_date}；预测标签: 下一交易日({next_trade_date})。")

        self.progress(30, "生成收益、量价、波动特征")
        g = df.groupby("ts_code", group_keys=False)
        df["ret1"] = df["pct_chg"] / 100.0
        df["gap"] = df["open"] / df["pre_close"].replace(0, np.nan) - 1.0
        df["intraday_ret"] = df["close"] / df["open"].replace(0, np.nan) - 1.0
        df["range_hl"] = df["high"] / df["low"].replace(0, np.nan) - 1.0
        df["close_pos"] = (df["close"] - df["low"]) / (df["high"] - df["low"]).replace(0, np.nan)

        for w in [3, 5, 10, 20, 60]:
            self._check_stop()
            df[f"ret_{w}"] = g["close"].pct_change(w)
            minp = max(2, w // 2)
            ma = fast_group_rolling(g, "close", w, minp, "mean")
            df[f"ma_gap_{w}"] = df["close"] / ma.replace(0, np.nan) - 1.0
            vol_ma = fast_group_rolling(g, "vol", w, minp, "mean")
            df[f"vol_ratio_{w}"] = df["vol"] / vol_ma.replace(0, np.nan)
            amt_ma = fast_group_rolling(g, "amount", w, minp, "mean")
            df[f"amount_ratio_{w}"] = df["amount"] / amt_ma.replace(0, np.nan)
            df[f"volatility_{w}"] = fast_group_rolling(g, "ret1", w, minp, "std")

        self.progress(38, "生成技术指标")
        df["roll_max_20"] = fast_group_rolling(g, "close", 20, 10, "max")
        df["roll_min_20"] = fast_group_rolling(g, "close", 20, 10, "min")
        df["to_20_high"] = df["close"] / df["roll_max_20"].replace(0, np.nan) - 1.0
        df["from_20_low"] = df["close"] / df["roll_min_20"].replace(0, np.nan) - 1.0
        df.drop(columns=["roll_max_20", "roll_min_20"], inplace=True, errors="ignore")

        # 量价形态：低位缩量上涨 / 高位放量下跌
        # 说明：
        # - price_position_w 越接近 0 越接近阶段低位，越接近 1 越接近阶段高位。
        # - low_volume_up_signal 是偏正向的潜伏/企稳信号。
        # - high_volume_down_signal 是偏风险的派发/出货信号。
        up_strength = (df["pct_chg"].clip(lower=0) / 5.0).clip(0, 2.0)
        down_strength = ((-df["pct_chg"]).clip(lower=0) / 5.0).clip(0, 2.0)
        low_scores = []
        high_risks = []
        for w in [20, 60]:
            minp = max(10, w // 2)
            roll_max = fast_group_rolling(g, "close", w, minp, "max")
            roll_min = fast_group_rolling(g, "close", w, minp, "min")
            pos = (df["close"] - roll_min) / (roll_max - roll_min).replace(0, np.nan)
            pos = pos.clip(0, 1)
            df[f"price_position_{w}"] = pos

            vol_ratio = df.get(f"vol_ratio_{w}", pd.Series(np.nan, index=df.index)).astype(float)
            amount_ratio = df.get(f"amount_ratio_{w}", pd.Series(np.nan, index=df.index)).astype(float)

            vol_shrink = ((self.cfg.vp_shrink_cutoff - vol_ratio) / max(self.cfg.vp_shrink_cutoff, 1e-6)).clip(0, 1.5)
            amt_shrink = ((self.cfg.vp_shrink_cutoff - amount_ratio) / max(self.cfg.vp_shrink_cutoff, 1e-6)).clip(0, 1.5)
            shrink_strength = pd.concat([vol_shrink, amt_shrink], axis=1).mean(axis=1, skipna=True).fillna(0)

            vol_expand = ((vol_ratio - self.cfg.vp_expand_cutoff) / max(self.cfg.vp_expand_cutoff, 1e-6)).clip(0, 2.0)
            amt_expand = ((amount_ratio - self.cfg.vp_expand_cutoff) / max(self.cfg.vp_expand_cutoff, 1e-6)).clip(0, 2.0)
            expand_strength = pd.concat([vol_expand, amt_expand], axis=1).mean(axis=1, skipna=True).fillna(0)

            low_strength = (1.0 - pos).clip(0, 1)
            high_strength = pos.clip(0, 1)

            df[f"low_volume_up_flag_{w}"] = ((pos <= self.cfg.vp_low_pos_cutoff) & (vol_ratio <= self.cfg.vp_shrink_cutoff) & (df["pct_chg"] > 0)).astype(float)
            df[f"high_volume_down_flag_{w}"] = ((pos >= self.cfg.vp_high_pos_cutoff) & (vol_ratio >= self.cfg.vp_expand_cutoff) & (df["pct_chg"] < 0)).astype(float)
            df[f"low_volume_up_score_{w}"] = low_strength * shrink_strength * up_strength
            df[f"high_volume_down_risk_{w}"] = high_strength * expand_strength * down_strength
            df[f"volume_pressure_balance_{w}"] = df[f"low_volume_up_score_{w}"] - df[f"high_volume_down_risk_{w}"]
            low_scores.append(df[f"low_volume_up_score_{w}"])
            high_risks.append(df[f"high_volume_down_risk_{w}"])

        df["low_volume_up_signal"] = pd.concat(low_scores, axis=1).mean(axis=1, skipna=True).fillna(0)
        df["high_volume_down_signal"] = pd.concat(high_risks, axis=1).mean(axis=1, skipna=True).fillna(0)
        df["volume_price_pattern_score"] = df["low_volume_up_signal"] - df["high_volume_down_signal"]

        # RSI14
        delta = g["close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.groupby(df["ts_code"]).transform(lambda s: s.rolling(14, min_periods=7).mean())
        avg_loss = loss.groupby(df["ts_code"]).transform(lambda s: s.rolling(14, min_periods=7).mean())
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df["rsi14"] = 100 - (100 / (1 + rs))

        # MACD
        ema12 = g["close"].transform(lambda s: s.ewm(span=12, adjust=False, min_periods=12).mean())
        ema26 = g["close"].transform(lambda s: s.ewm(span=26, adjust=False, min_periods=26).mean())
        df["macd_dif"] = ema12 - ema26
        df["macd_dea"] = df.groupby("ts_code")["macd_dif"].transform(lambda s: s.ewm(span=9, adjust=False, min_periods=9).mean())
        df["macd_hist"] = df["macd_dif"] - df["macd_dea"]

        self.progress(45, "生成资金流与市值特征")
        money_cols = [c for c in df.columns if any(k in c for k in ["buy_", "sell_", "net_mf_"])]
        for c in money_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        if "net_mf_amount" in df.columns and "amount" in df.columns:
            df["net_mf_amount_ratio"] = df["net_mf_amount"] / df["amount"].replace(0, np.nan)
        if "buy_lg_amount" in df.columns and "sell_lg_amount" in df.columns and "amount" in df.columns:
            df["large_order_net_ratio"] = (df["buy_lg_amount"] - df["sell_lg_amount"]) / df["amount"].replace(0, np.nan)
        if "buy_elg_amount" in df.columns and "sell_elg_amount" in df.columns and "amount" in df.columns:
            df["elarge_order_net_ratio"] = (df["buy_elg_amount"] - df["sell_elg_amount"]) / df["amount"].replace(0, np.nan)

        # 类别编码，不使用 one-hot，避免字段爆炸
        for c in ["industry", "market", "area"]:
            if c in df.columns:
                df[c] = df[c].fillna("未知").astype(str)
                df[f"{c}_code"] = pd.Categorical(df[c]).codes.astype(float)

        if "list_date" in df.columns:
            td = pd.to_datetime(df["trade_date"], format="%Y%m%d", errors="coerce")
            ld = pd.to_datetime(df["list_date"].astype(str), format="%Y%m%d", errors="coerce")
            df["days_listed"] = (td - ld).dt.days.astype("float")

        self.progress(52, "生成次日标签")
        df["next_pct_chg"] = g["pct_chg"].shift(-1)
        df["y_limit"] = (df["next_pct_chg"] >= self.cfg.limit_threshold).astype(float)
        df["y_bigrise"] = (df["next_pct_chg"] >= self.cfg.bigrise_threshold).astype(float)
        df.loc[df["next_pct_chg"].isna(), ["y_limit", "y_bigrise"]] = np.nan

        id_cols = {
            "ts_code", "trade_date", "name", "industry", "market", "area", "list_date",
            "next_pct_chg", "y_limit", "y_bigrise"
        }
        feature_cols = []
        for c in df.columns:
            if c in id_cols:
                continue
            if pd.api.types.is_numeric_dtype(df[c]):
                feature_cols.append(c)

        # 删除全空/几乎常数特征
        good_cols = []
        for c in feature_cols:
            s = df[c]
            if s.notna().sum() < 100:
                continue
            if s.nunique(dropna=True) <= 1:
                continue
            good_cols.append(c)
        feature_cols = good_cols
        if not feature_cols:
            raise RuntimeError("没有可用特征列，请检查 daily/daily_basic/moneyflow 表字段。")
        self.log(f"最终特征数: {len(feature_cols)}。")
        return df, feature_cols, latest_trade_date, next_trade_date

    def _guess_next_trade_date(self, latest: str) -> str:
        # 如果 trade_calendar 表中有未来开市日，则取下一个；否则只返回“下一交易日”。
        try:
            conn = connect_db(self.cfg.db_path)
            if table_exists(conn, "trade_calendar"):
                cols = get_table_cols(conn, "trade_calendar")
                if "cal_date" in cols and "is_open" in cols:
                    q = pd.read_sql(
                        'SELECT cal_date FROM trade_calendar WHERE is_open=1 AND cal_date>? ORDER BY cal_date LIMIT 1',
                        conn,
                        params=(latest,),
                    )
                    if not q.empty:
                        return str(q.loc[0, "cal_date"])
        except Exception:
            pass
        finally:
            try:
                conn.close()
            except Exception:
                pass
        return "下一交易日"


# -----------------------------
# 模型工具：随机森林/ExtraTrees
# -----------------------------

def _sample_training_rows(df_train: pd.DataFrame, target: str, max_rows: int, seed: int) -> pd.DataFrame:
    if len(df_train) <= max_rows:
        return df_train
    rng = np.random.default_rng(seed)
    pos = df_train[df_train[target] == 1]
    neg = df_train[df_train[target] == 0]
    # 保留全部正例，负例随机采样；正例特别多时再整体采样
    if len(pos) >= max_rows * 0.45:
        return df_train.sample(max_rows, random_state=seed)
    neg_n = max_rows - len(pos)
    if neg_n <= 0:
        return pos.sample(max_rows, random_state=seed)
    neg_idx = rng.choice(neg.index.to_numpy(), size=min(neg_n, len(neg)), replace=False)
    out = pd.concat([pos, neg.loc[neg_idx]], axis=0).sample(frac=1, random_state=seed)
    return out


class TabularBinaryModel:
    def __init__(self, cfg: AppConfig, target: str, log: Callable[[str], None]):
        self.cfg = cfg
        self.target = target
        self.log = log
        self.imputer = SimpleImputer(strategy="median")
        self.model = None
        self.feature_cols: List[str] = []
        self.backend_used = "sklearn"
        self.classes_ = np.array([0, 1])

    def _positive_prior(self, y: np.ndarray) -> float:
        return float(np.mean(y)) if len(y) else 0.0

    def _to_numpy_probability(self, prob_obj, n: int) -> np.ndarray:
        try:
            if hasattr(prob_obj, "to_numpy"):
                prob = prob_obj.to_numpy()
            elif hasattr(prob_obj, "get"):
                prob = prob_obj.get()
            else:
                prob = np.asarray(prob_obj)
        except Exception:
            prob = np.asarray(prob_obj)
        prob = np.asarray(prob)
        if prob.ndim == 1:
            return prob.astype(float)
        if prob.shape[1] == 1:
            cls = int(self.classes_[0]) if len(self.classes_) else 0
            return np.ones(n, dtype=float) if cls == 1 else np.zeros(n, dtype=float)
        pos_idx = list(self.classes_).index(1) if 1 in list(self.classes_) else prob.shape[1] - 1
        return prob[:, pos_idx].astype(float)

    def _fit_cuml(self, X_imp: np.ndarray, y: np.ndarray) -> bool:
        if self.cfg.tree_model == "ExtraTrees":
            return False
        try:
            import cudf
            from cuml.ensemble import RandomForestClassifier as CuMLRandomForestClassifier
            X_gpu = cudf.DataFrame(X_imp, columns=self.feature_cols)
            y_gpu = cudf.Series(y.astype(np.int32))
            max_depth = 16 if self.cfg.rf_max_depth <= 0 else int(self.cfg.rf_max_depth)
            self.model = CuMLRandomForestClassifier(
                n_estimators=int(self.cfg.rf_n_estimators),
                max_depth=max_depth,
                random_state=int(self.cfg.rf_random_state),
                n_streams=8,
            )
            self.model.fit(X_gpu, y_gpu)
            self.backend_used = "cuml"
            self.classes_ = np.array([0, 1])
            self.log(f"cuML GPU RandomForest 已启用 - {self.target}: cuda 优先，rows={len(y):,}, pos={int(y.sum()):,}")
            return True
        except Exception as e:
            self.log(f"cuML GPU 树模型不可用，尝试下一个后端：{e}")
            self.model = None
            return False

    def _fit_xgboost_gpu(self, X_imp: np.ndarray, y: np.ndarray) -> bool:
        try:
            from xgboost import XGBClassifier
            pos = max(float(y.sum()), 1.0)
            neg = max(float(len(y) - y.sum()), 1.0)
            depth = 8 if self.cfg.rf_max_depth <= 0 else int(self.cfg.rf_max_depth)
            lr = 0.045 if int(self.cfg.rf_n_estimators) >= 400 else 0.06
            device = f"cuda:{int(getattr(self.cfg, 'preferred_primary_gpu', 0))}"
            self.model = XGBClassifier(
                n_estimators=int(self.cfg.rf_n_estimators),
                max_depth=depth,
                learning_rate=lr,
                subsample=0.85,
                colsample_bytree=0.85,
                min_child_weight=max(1.0, float(self.cfg.rf_min_samples_leaf) / 4.0),
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                device=device,
                random_state=int(self.cfg.rf_random_state),
                n_jobs=1,
                scale_pos_weight=(neg / pos) if self.cfg.rf_class_weight else 1.0,
            )
            self.model.fit(X_imp, y)
            self.backend_used = "xgboost_gpu"
            self.classes_ = np.array([0, 1])
            self.log(f"XGBoost GPU 树模型已启用 - {self.target}: {device}, rows={len(y):,}, pos={int(y.sum()):,}")
            return True
        except Exception as e:
            self.log(f"XGBoost GPU 树模型不可用，回退到 sklearn CPU：{e}")
            self.model = None
            return False

    def fit(self, df_train: pd.DataFrame, feature_cols: List[str]):
        self.feature_cols = feature_cols
        train_use = _sample_training_rows(df_train, self.target, self.cfg.max_train_rows, self.cfg.rf_random_state)
        y = train_use[self.target].astype(int).to_numpy()
        X = train_use[feature_cols].replace([np.inf, -np.inf], np.nan)
        X_imp = self.imputer.fit_transform(X).astype(np.float32, copy=False)
        self.classes_ = np.unique(y)
        backend = str(getattr(self.cfg, "tree_backend", "auto") or "auto").lower()
        if backend in {"auto", "cuml", "rapids", "gpu"} and self._fit_cuml(X_imp, y):
            return self
        if backend in {"auto", "xgboost", "xgboost_gpu", "gpu"} and self._fit_xgboost_gpu(X_imp, y):
            return self
        if backend in {"cuml", "rapids"}:
            self.log("指定了 cuML 后端但不可用；为保证任务完成，已回退 sklearn CPU。")
        if backend in {"xgboost", "xgboost_gpu"}:
            self.log("指定了 XGBoost GPU 后端但不可用；为保证任务完成，已回退 sklearn CPU。")
        class_weight = "balanced_subsample" if self.cfg.rf_class_weight else None
        if self.cfg.tree_model == "ExtraTrees":
            self.model = ExtraTreesClassifier(
                n_estimators=self.cfg.rf_n_estimators,
                max_depth=None if self.cfg.rf_max_depth <= 0 else self.cfg.rf_max_depth,
                min_samples_leaf=self.cfg.rf_min_samples_leaf,
                n_jobs=self.cfg.rf_n_jobs,
                random_state=self.cfg.rf_random_state,
                class_weight=class_weight,
                bootstrap=False,
            )
        else:
            self.model = RandomForestClassifier(
                n_estimators=self.cfg.rf_n_estimators,
                max_depth=None if self.cfg.rf_max_depth <= 0 else self.cfg.rf_max_depth,
                min_samples_leaf=self.cfg.rf_min_samples_leaf,
                n_jobs=self.cfg.rf_n_jobs,
                random_state=self.cfg.rf_random_state,
                class_weight=class_weight,
                bootstrap=True,
                max_samples=0.85,
            )
        self.backend_used = "sklearn"
        self.log(f"训练 sklearn {self.cfg.tree_model} - {self.target}: rows={len(train_use):,}, pos={int(y.sum()):,}")
        self.model.fit(X_imp, y)
        self.classes_ = getattr(self.model, "classes_", np.unique(y))
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("模型尚未训练。")
        X = df[self.feature_cols].replace([np.inf, -np.inf], np.nan)
        X_imp = self.imputer.transform(X).astype(np.float32, copy=False)
        if self.backend_used == "cuml":
            try:
                import cudf
                X_gpu = cudf.DataFrame(X_imp, columns=self.feature_cols)
                return self._to_numpy_probability(self.model.predict_proba(X_gpu), len(df))
            except Exception as e:
                self.log(f"cuML predict_proba 失败，尝试直接 numpy 预测：{e}")
        if self.backend_used == "xgboost_gpu":
            return self._to_numpy_probability(self.model.predict_proba(X_imp), len(df))
        prob = self.model.predict_proba(X_imp)
        if prob.shape[1] == 1:
            # 极端情况下训练集只有一类
            cls = int(self.classes_[0])
            return np.ones(len(df)) if cls == 1 else np.zeros(len(df))
        pos_idx = list(self.classes_).index(1)
        return prob[:, pos_idx]

    def validate(self, df_val: pd.DataFrame) -> Dict[str, float]:
        y = df_val[self.target].astype(int).to_numpy()
        p = self.predict_proba(df_val)
        out = {}
        if len(np.unique(y)) > 1:
            out["auc"] = float(roc_auc_score(y, p))
            out["ap"] = float(average_precision_score(y, p))
        out["pos_rate"] = float(y.mean()) if len(y) else np.nan
        return out


# -----------------------------
# 模型工具：液态神经网络/LTC 风格 PyTorch 实现
# -----------------------------

class TorchUnavailableError(RuntimeError):
    pass


def import_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch.utils.data import DataLoader, TensorDataset
        return torch, nn, F, DataLoader, TensorDataset
    except Exception as e:
        raise TorchUnavailableError("LNN 需要 PyTorch。请安装：python -m pip install -U torch") from e


def resolve_cuda_devices(cfg: AppConfig, log: Callable[[str], None]) -> Tuple[str, List[int], List[str]]:
    torch, _, _, _, _ = import_torch()
    if not torch.cuda.is_available():
        if cfg.allow_cpu_fallback:
            log("未检测到可用 CUDA，LNN 使用 CPU。")
            return "cpu", [], []
        raise RuntimeError("未检测到 CUDA，且未允许 CPU fallback。")

    device_names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    selected = []
    preferred = [int(getattr(cfg, "preferred_primary_gpu", 0)), int(getattr(cfg, "preferred_secondary_gpu", 1))]
    for i in preferred:
        if 0 <= i < len(device_names) and i not in selected:
            lname = device_names[i].lower()
            if (cfg.use_5090d and "5090" in lname) or (cfg.use_4090d and "4090" in lname):
                selected.append(i)
    for i, name in enumerate(device_names):
        lname = name.lower()
        if cfg.use_5090d and "5090" in lname and i not in selected:
            selected.append(i)
        if cfg.use_4090d and "4090" in lname and i not in selected:
            selected.append(i)
    if not bool(getattr(cfg, "use_dual_gpu_parallel", True)) and selected:
        selected = selected[:1]
    if not selected:
        if cfg.use_4090d or cfg.use_5090d:
            log(f"没有匹配到勾选的 4090/5090，检测到 GPU: {device_names}。默认使用 cuda:0。")
        selected = [0]
    log("LNN 使用 GPU: " + "; ".join([f"cuda:{i} {device_names[i]}" for i in selected]))
    if len(selected) > 1:
        log(f"双 GPU 并行已启用：主卡 cuda:{selected[0]}，协同卡 {', '.join('cuda:'+str(i) for i in selected[1:])}。")
    return f"cuda:{selected[0]}", selected, device_names


def build_lnn_classes():
    torch, nn, F, DataLoader, TensorDataset = import_torch()

    class LiquidCell(nn.Module):
        """工程版 Liquid Time-Constant 风格循环单元。

        h_{t+1} = h_t + dt * (-h_t + tanh(Wx_t + Uh_t + b)) / tau
        tau 通过 softplus 保证为正。
        """
        def __init__(self, input_size: int, hidden_size: int, dropout: float = 0.1):
            super().__init__()
            self.input = nn.Linear(input_size, hidden_size)
            self.recurrent = nn.Linear(hidden_size, hidden_size, bias=False)
            self.log_tau = nn.Parameter(torch.zeros(hidden_size))
            self.dropout = nn.Dropout(dropout)

        def forward(self, x, h):
            tau = F.softplus(self.log_tau) + 0.1
            drive = torch.tanh(self.input(x) + self.recurrent(h))
            dh = (-h + drive) / tau
            h = h + dh
            return self.dropout(h)

    class LNNClassifier(nn.Module):
        def __init__(self, input_size: int, hidden_size: int, dropout: float = 0.1):
            super().__init__()
            self.hidden_size = hidden_size
            self.cell = LiquidCell(input_size, hidden_size, dropout)
            self.norm = nn.LayerNorm(hidden_size)
            self.head = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 2, 1),
            )

        def forward(self, x):
            # x: [B, T, F]
            b = x.size(0)
            h = torch.zeros(b, self.hidden_size, dtype=x.dtype, device=x.device)
            for t in range(x.size(1)):
                h = self.cell(x[:, t, :], h)
            h = self.norm(h)
            return self.head(h).squeeze(-1)

    return torch, nn, F, DataLoader, TensorDataset, LNNClassifier


def fit_feature_scaler(df_train: pd.DataFrame, feature_cols: List[str]):
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X = df_train[feature_cols].replace([np.inf, -np.inf], np.nan)
    X_imp = imputer.fit_transform(X)
    X_scaled = scaler.fit_transform(X_imp).astype(np.float32)
    return imputer, scaler, X_scaled


def transform_features(df: pd.DataFrame, feature_cols: List[str], imputer: SimpleImputer, scaler: StandardScaler) -> np.ndarray:
    X = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    X_imp = imputer.transform(X)
    return scaler.transform(X_imp).astype(np.float32)


def build_train_sequences(
    df_train: pd.DataFrame,
    X_scaled: np.ndarray,
    target: str,
    seq_len: int,
    max_rows: int,
    seed: int,
    log: Callable[[str], None],
    stop_event: threading.Event,
) -> Tuple[np.ndarray, np.ndarray]:
    local = df_train[["ts_code", "trade_date", target]].reset_index(drop=True).copy()
    local["arr_idx"] = np.arange(len(local))
    local = local.sort_values(["ts_code", "trade_date"])
    seqs = []
    ys = []
    for _, sub in local.groupby("ts_code", sort=False):
        if stop_event.is_set():
            raise RuntimeError("用户已停止任务。")
        idx = sub["arr_idx"].to_numpy()
        yv = sub[target].to_numpy()
        if len(idx) < seq_len:
            continue
        for j in range(seq_len - 1, len(idx)):
            if not np.isfinite(yv[j]):
                continue
            seqs.append(idx[j - seq_len + 1:j + 1])
            ys.append(int(yv[j]))
    if not seqs:
        raise RuntimeError("LNN 没有构造出训练序列，请降低 seq_len 或扩大训练时间。")

    ys = np.asarray(ys, dtype=np.int64)
    seq_idx = np.asarray(seqs, dtype=np.int64)

    if len(seq_idx) > max_rows:
        rng = np.random.default_rng(seed)
        pos = np.where(ys == 1)[0]
        neg = np.where(ys == 0)[0]
        if len(pos) >= max_rows * 0.45:
            keep = rng.choice(np.arange(len(seq_idx)), size=max_rows, replace=False)
        else:
            neg_n = max_rows - len(pos)
            neg_keep = rng.choice(neg, size=min(neg_n, len(neg)), replace=False)
            keep = np.concatenate([pos, neg_keep])
            rng.shuffle(keep)
        seq_idx = seq_idx[keep]
        ys = ys[keep]

    X_seq = X_scaled[seq_idx]
    log(f"LNN 序列样本: {len(X_seq):,}, seq_len={seq_len}, pos={int(ys.sum()):,}")
    return X_seq.astype(np.float32), ys.astype(np.float32)


def build_latest_sequences(
    df_all: pd.DataFrame,
    X_all_scaled: np.ndarray,
    latest_trade_date: str,
    seq_len: int,
    stop_event: threading.Event,
) -> Tuple[List[str], np.ndarray]:
    local = df_all[["ts_code", "trade_date"]].reset_index(drop=True).copy()
    local["arr_idx"] = np.arange(len(local))
    local = local.sort_values(["ts_code", "trade_date"])
    codes = []
    seqs = []
    for code, sub in local.groupby("ts_code", sort=False):
        if stop_event.is_set():
            raise RuntimeError("用户已停止任务。")
        sub = sub.reset_index(drop=True)
        hit = sub.index[sub["trade_date"].astype(str) == str(latest_trade_date)].tolist()
        if not hit:
            continue
        j = hit[-1]
        if j < seq_len - 1:
            continue
        idx = sub.loc[j - seq_len + 1:j, "arr_idx"].to_numpy()
        codes.append(str(code))
        seqs.append(idx)
    if not seqs:
        return [], np.empty((0, seq_len, X_all_scaled.shape[1]), dtype=np.float32)
    seq_idx = np.asarray(seqs, dtype=np.int64)
    return codes, X_all_scaled[seq_idx].astype(np.float32)


class LNNBinaryModel:
    def __init__(self, cfg: AppConfig, target: str, feature_cols: List[str], log: Callable[[str], None], stop_event: threading.Event):
        self.cfg = cfg
        self.target = target
        self.feature_cols = feature_cols
        self.log = log
        self.stop_event = stop_event
        self.imputer: Optional[SimpleImputer] = None
        self.scaler: Optional[StandardScaler] = None
        self.model = None
        self.device = "cpu"
        self.device_ids: List[int] = []

    def fit(self, df_train: pd.DataFrame):
        torch, nn, F, DataLoader, TensorDataset, LNNClassifier = build_lnn_classes()
        random.seed(self.cfg.lnn_random_state)
        np.random.seed(self.cfg.lnn_random_state)
        torch.manual_seed(self.cfg.lnn_random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.cfg.lnn_random_state)

        self.device, self.device_ids, _ = resolve_cuda_devices(self.cfg, self.log)
        self.imputer, self.scaler, X_scaled = fit_feature_scaler(df_train, self.feature_cols)
        X_seq, y_seq = build_train_sequences(
            df_train=df_train,
            X_scaled=X_scaled,
            target=self.target,
            seq_len=self.cfg.lnn_seq_len,
            max_rows=self.cfg.max_train_rows,
            seed=self.cfg.lnn_random_state,
            log=self.log,
            stop_event=self.stop_event,
        )

        ds = TensorDataset(torch.from_numpy(X_seq), torch.from_numpy(y_seq))
        pin_memory = str(self.device).startswith("cuda")
        loader = DataLoader(ds, batch_size=self.cfg.lnn_batch_size, shuffle=True, num_workers=0, drop_last=False, pin_memory=pin_memory)
        model = LNNClassifier(input_size=len(self.feature_cols), hidden_size=self.cfg.lnn_hidden_size, dropout=self.cfg.lnn_dropout)
        if bool(getattr(self.cfg, "use_model_registry", True)) and bool(getattr(self.cfg, "continue_lnn_from_latest", True)):
            try:
                registry = ModelRegistry(getattr(self.cfg, "model_registry_dir", r"model_registry"), self.log)
                expected = {
                    "input_size": len(self.feature_cols),
                    "hidden_size": int(self.cfg.lnn_hidden_size),
                    "seq_len": int(self.cfg.lnn_seq_len),
                    "dropout": float(self.cfg.lnn_dropout),
                }
                entry = registry.compatible_latest("lnn", self.target, self.feature_cols, expected)
                if entry:
                    state = torch.load(entry["weight_path"], map_location="cpu")
                    model.load_state_dict(state, strict=True)
                    self.log(f"LNN {self.target} 已加载上一版权重继续训练：{entry['weight_path']}")
            except Exception as e:
                self.log(f"LNN {self.target} 读取上一版权重失败，改为重新训练：{e}")
        model.to(self.device)
        if len(self.device_ids) > 1:
            model = nn.DataParallel(model, device_ids=self.device_ids)

        pos = float(y_seq.sum())
        neg = float(len(y_seq) - pos)
        pos_weight = torch.tensor([max(neg / max(pos, 1.0), 1.0)], dtype=torch.float32, device=self.device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        opt = torch.optim.AdamW(model.parameters(), lr=self.cfg.lnn_lr, weight_decay=1e-4)
        use_amp = str(self.device).startswith("cuda")
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        else:
            scaler = torch.cuda.amp.GradScaler(enabled=use_amp) if hasattr(torch.cuda, "amp") else None

        self.log(f"训练 LNN - {self.target}: epochs={self.cfg.lnn_epochs}, hidden={self.cfg.lnn_hidden_size}, device={self.device}")
        model.train()
        last_loss = 0.0
        for ep in range(1, self.cfg.lnn_epochs + 1):
            if self.stop_event.is_set():
                raise RuntimeError("用户已停止任务。")
            total_loss = 0.0
            n = 0
            for xb, yb in loader:
                xb = xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                if use_amp and scaler is not None:
                    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
                        autocast_ctx = torch.amp.autocast("cuda", enabled=True)
                    else:
                        autocast_ctx = torch.cuda.amp.autocast(enabled=True)
                    with autocast_ctx:
                        logits = model(xb)
                        loss = loss_fn(logits, yb)
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    scaler.step(opt)
                    scaler.update()
                else:
                    logits = model(xb)
                    loss = loss_fn(logits, yb)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    opt.step()
                bs = xb.size(0)
                total_loss += float(loss.item()) * bs
                n += bs
            last_loss = total_loss / max(n, 1)
            self.log(f"LNN {self.target} epoch {ep}/{self.cfg.lnn_epochs}, loss={last_loss:.6f}")
        self.model = model
        if bool(getattr(self.cfg, "use_model_registry", True)):
            try:
                registry = ModelRegistry(getattr(self.cfg, "model_registry_dir", r"model_registry"), self.log)
                metrics = {
                    "last_loss": float(last_loss),
                    "quality_metric": float(1.0 / (1.0 + max(float(last_loss), 0.0))),
                    "sample_count": float(len(y_seq)),
                    "positive_count": float(y_seq.sum()),
                }
                registry.save_torch_model(
                    model=model,
                    model_kind="lnn",
                    target=self.target,
                    cfg=self.cfg,
                    feature_cols=self.feature_cols,
                    arch_meta={
                        "input_size": len(self.feature_cols),
                        "hidden_size": int(self.cfg.lnn_hidden_size),
                        "seq_len": int(self.cfg.lnn_seq_len),
                        "dropout": float(self.cfg.lnn_dropout),
                    },
                    metrics=metrics,
                )
            except Exception as e:
                self.log(f"LNN {self.target} 保存 model_registry 失败：{e}")
        return self

    def predict_latest(self, df_all: pd.DataFrame, latest_trade_date: str) -> Dict[str, float]:
        if self.model is None or self.imputer is None or self.scaler is None:
            raise RuntimeError("LNN 模型尚未训练。")
        torch, _, _, DataLoader, TensorDataset, _ = build_lnn_classes()
        df_all_local = df_all.reset_index(drop=True)
        X_all = transform_features(df_all_local, self.feature_cols, self.imputer, self.scaler)
        codes, X_seq = build_latest_sequences(df_all_local, X_all, latest_trade_date, self.cfg.lnn_seq_len, self.stop_event)
        if len(codes) == 0:
            return {}
        ds = TensorDataset(torch.from_numpy(X_seq))
        pin_memory = str(self.device).startswith("cuda")
        loader = DataLoader(ds, batch_size=max(self.cfg.lnn_batch_size * 2, 1024), shuffle=False, num_workers=0, pin_memory=pin_memory)
        self.model.eval()
        probs = []
        with torch.no_grad():
            for (xb,) in loader:
                xb = xb.to(self.device, non_blocking=True)
                if str(self.device).startswith("cuda") and (hasattr(torch, "amp") or hasattr(torch.cuda, "amp")):
                    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
                        autocast_ctx = torch.amp.autocast("cuda", enabled=True)
                    else:
                        autocast_ctx = torch.cuda.amp.autocast(enabled=True)
                    with autocast_ctx:
                        logits = self.model(xb)
                else:
                    logits = self.model(xb)
                p = torch.sigmoid(logits).detach().cpu().numpy()
                probs.append(p)
        p_all = np.concatenate(probs, axis=0)
        return {code: float(p) for code, p in zip(codes, p_all)}


def build_transformer_classes():
    torch, nn, F, DataLoader, TensorDataset = import_torch()

    class TimeSeriesTransformerClassifier(nn.Module):
        def __init__(self, input_size: int, d_model: int, nhead: int, num_layers: int, dropout: float = 0.1):
            super().__init__()
            self.input_size = input_size
            self.d_model = d_model
            self.proj = nn.Linear(input_size, d_model)
            layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=max(d_model * 4, 128),
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
            self.norm = nn.LayerNorm(d_model)
            self.head = nn.Sequential(
                nn.Linear(d_model, max(d_model // 2, 32)),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(max(d_model // 2, 32), 1),
            )

        def forward(self, x):
            z = self.proj(x)
            z = self.encoder(z)
            z = self.norm(z[:, -1, :])
            return self.head(z).squeeze(-1)

    return torch, nn, F, DataLoader, TensorDataset, TimeSeriesTransformerClassifier


class TransformerBinaryModel:
    def __init__(self, cfg: AppConfig, target: str, feature_cols: List[str], log: Callable[[str], None], stop_event: threading.Event):
        self.cfg = cfg
        self.target = target
        self.feature_cols = feature_cols
        self.log = log
        self.stop_event = stop_event
        self.imputer: Optional[SimpleImputer] = None
        self.scaler: Optional[StandardScaler] = None
        self.model = None
        self.device = "cpu"
        self.device_ids: List[int] = []

    def fit(self, df_train: pd.DataFrame):
        torch, nn, F, DataLoader, TensorDataset, TransformerClassifier = build_transformer_classes()
        seed = int(getattr(self.cfg, "lnn_random_state", 2026)) + 707
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        self.device, self.device_ids, _ = resolve_cuda_devices(self.cfg, self.log)
        self.imputer, self.scaler, X_scaled = fit_feature_scaler(df_train, self.feature_cols)
        seq_len = int(getattr(self.cfg, "transformer_seq_len", 32) or 32)
        X_seq, y_seq = build_train_sequences(
            df_train=df_train,
            X_scaled=X_scaled,
            target=self.target,
            seq_len=seq_len,
            max_rows=int(getattr(self.cfg, "max_train_rows", 250000)),
            seed=seed,
            log=self.log,
            stop_event=self.stop_event,
        )

        ds = TensorDataset(torch.from_numpy(X_seq), torch.from_numpy(y_seq))
        pin_memory = str(self.device).startswith("cuda")
        batch_size = int(getattr(self.cfg, "transformer_batch_size", 768) or 768)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False, pin_memory=pin_memory)
        d_model = int(getattr(self.cfg, "transformer_d_model", 128) or 128)
        heads = int(getattr(self.cfg, "transformer_heads", 4) or 4)
        layers = int(getattr(self.cfg, "transformer_layers", 2) or 2)
        dropout = float(getattr(self.cfg, "transformer_dropout", 0.12) or 0.12)
        model = TransformerClassifier(
            input_size=len(self.feature_cols),
            d_model=d_model,
            nhead=heads,
            num_layers=layers,
            dropout=dropout,
        )

        if bool(getattr(self.cfg, "use_model_registry", True)) and bool(getattr(self.cfg, "continue_transformer_from_latest", True)):
            try:
                registry = ModelRegistry(getattr(self.cfg, "model_registry_dir", r"model_registry"), self.log)
                expected = {
                    "input_size": len(self.feature_cols),
                    "d_model": d_model,
                    "heads": heads,
                    "layers": layers,
                    "seq_len": seq_len,
                    "dropout": dropout,
                }
                entry = registry.compatible_latest("transformer", self.target, self.feature_cols, expected)
                if entry:
                    state = torch.load(entry["weight_path"], map_location="cpu")
                    model.load_state_dict(state, strict=True)
                    self.log(f"Transformer {self.target} 已加载上一版权重继续训练：{entry['weight_path']}")
            except Exception as e:
                self.log(f"Transformer {self.target} 读取上一版权重失败，改为重新训练：{e}")

        model.to(self.device)
        if len(self.device_ids) > 1:
            model = nn.DataParallel(model, device_ids=self.device_ids)

        pos = float(y_seq.sum())
        neg = float(len(y_seq) - pos)
        pos_weight = torch.tensor([max(neg / max(pos, 1.0), 1.0)], dtype=torch.float32, device=self.device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        opt = torch.optim.AdamW(model.parameters(), lr=float(getattr(self.cfg, "transformer_lr", 8e-4)), weight_decay=1e-4)
        use_amp = str(self.device).startswith("cuda")
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        else:
            scaler = torch.cuda.amp.GradScaler(enabled=use_amp) if hasattr(torch.cuda, "amp") else None

        epochs = int(getattr(self.cfg, "transformer_epochs", 6) or 6)
        self.log(f"训练 Transformer - {self.target}: epochs={epochs}, d_model={d_model}, layers={layers}, heads={heads}, device={self.device}")
        model.train()
        last_loss = 0.0
        for ep in range(1, epochs + 1):
            if self.stop_event.is_set():
                raise RuntimeError("用户已停止任务。")
            total_loss = 0.0
            n = 0
            for xb, yb in loader:
                xb = xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                if use_amp and scaler is not None:
                    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
                        autocast_ctx = torch.amp.autocast("cuda", enabled=True)
                    else:
                        autocast_ctx = torch.cuda.amp.autocast(enabled=True)
                    with autocast_ctx:
                        logits = model(xb)
                        loss = loss_fn(logits, yb)
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    scaler.step(opt)
                    scaler.update()
                else:
                    logits = model(xb)
                    loss = loss_fn(logits, yb)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    opt.step()
                bs = xb.size(0)
                total_loss += float(loss.item()) * bs
                n += bs
            last_loss = total_loss / max(n, 1)
            self.log(f"Transformer {self.target} epoch {ep}/{epochs}, loss={last_loss:.6f}")
        self.model = model

        if bool(getattr(self.cfg, "use_model_registry", True)):
            try:
                registry = ModelRegistry(getattr(self.cfg, "model_registry_dir", r"model_registry"), self.log)
                metrics = {
                    "last_loss": float(last_loss),
                    "quality_metric": float(1.0 / (1.0 + max(float(last_loss), 0.0))),
                    "sample_count": float(len(y_seq)),
                    "positive_count": float(y_seq.sum()),
                }
                registry.save_torch_model(
                    model=model,
                    model_kind="transformer",
                    target=self.target,
                    cfg=self.cfg,
                    feature_cols=self.feature_cols,
                    arch_meta={
                        "input_size": len(self.feature_cols),
                        "d_model": d_model,
                        "heads": heads,
                        "layers": layers,
                        "seq_len": seq_len,
                        "dropout": dropout,
                    },
                    metrics=metrics,
                )
            except Exception as e:
                self.log(f"Transformer {self.target} 保存 model_registry 失败：{e}")
        return self

    def predict_latest(self, df_all: pd.DataFrame, latest_trade_date: str) -> Dict[str, float]:
        if self.model is None or self.imputer is None or self.scaler is None:
            raise RuntimeError("Transformer 模型尚未训练。")
        torch, _, _, DataLoader, TensorDataset, _ = build_transformer_classes()
        df_all_local = df_all.reset_index(drop=True)
        X_all = transform_features(df_all_local, self.feature_cols, self.imputer, self.scaler)
        seq_len = int(getattr(self.cfg, "transformer_seq_len", 32) or 32)
        codes, X_seq = build_latest_sequences(df_all_local, X_all, latest_trade_date, seq_len, self.stop_event)
        if len(codes) == 0:
            return {}
        ds = TensorDataset(torch.from_numpy(X_seq))
        pin_memory = str(self.device).startswith("cuda")
        loader = DataLoader(ds, batch_size=max(int(getattr(self.cfg, "transformer_batch_size", 768)) * 2, 1024), shuffle=False, num_workers=0, pin_memory=pin_memory)
        self.model.eval()
        probs = []
        with torch.no_grad():
            for (xb,) in loader:
                xb = xb.to(self.device, non_blocking=True)
                if str(self.device).startswith("cuda") and (hasattr(torch, "amp") or hasattr(torch.cuda, "amp")):
                    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
                        autocast_ctx = torch.amp.autocast("cuda", enabled=True)
                    else:
                        autocast_ctx = torch.cuda.amp.autocast(enabled=True)
                    with autocast_ctx:
                        logits = self.model(xb)
                else:
                    logits = self.model(xb)
                p = torch.sigmoid(logits).detach().cpu().numpy()
                probs.append(p)
        p_all = np.concatenate(probs, axis=0)
        return {code: float(p) for code, p in zip(codes, p_all)}


# -----------------------------
# 预测引擎
# -----------------------------

class PredictionEngine:
    def __init__(self, cfg: AppConfig, log: Callable[[str], None], progress: Callable[[int, str], None], stop_event: threading.Event):
        self.cfg = cfg
        self.log = log
        self.progress = progress
        self.stop_event = stop_event

    def _check_stop(self):
        if self.stop_event.is_set():
            raise RuntimeError("用户已停止任务。")

    # ------------------------------------------------------------------
    # 马尔可夫-贝叶斯式三日回测自动选参
    # ------------------------------------------------------------------
    def _make_param_candidates(self) -> List[Dict[str, object]]:
        """
        生成少量可解释的参数组合。
        不做暴力网格搜索，避免第一次运行过慢；每个组合都围绕“状态转移概率 + 证据修正”展开。
        """
        c = self.cfg

        def norm_weights(w_limit: float, w_big: float) -> Tuple[float, float]:
            s = max(w_limit + w_big, 1e-9)
            wl, wb = w_limit / s, w_big / s
            cap = float(getattr(c, "max_auto_weight_limit", 0.70))
            cap = min(max(cap, 0.50), 0.95)
            if wl > cap:
                wl, wb = cap, 1.0 - cap
            return wl, wb

        raw_specs = []

        def add(name: str, tree_model=None, n=None, depth=None, leaf=None, topn=None,
                w_limit=None, w_big=None, bonus=None, penalty=None):
            wl, wb = norm_weights(
                c.weight_limit if w_limit is None else float(w_limit),
                c.weight_bigrise if w_big is None else float(w_big),
            )
            raw_specs.append({
                "candidate_name": name,
                "tree_model": tree_model or c.tree_model,
                "rf_n_estimators": int(n if n is not None else c.rf_n_estimators),
                "rf_max_depth": int(depth if depth is not None else c.rf_max_depth),
                "rf_min_samples_leaf": int(leaf if leaf is not None else c.rf_min_samples_leaf),
                "first_stage_topn": int(topn if topn is not None else c.first_stage_topn),
                "weight_limit": wl,
                "weight_bigrise": wb,
                "low_volume_up_bonus": float(bonus if bonus is not None else c.low_volume_up_bonus),
                "high_volume_down_penalty": float(penalty if penalty is not None else c.high_volume_down_penalty),
            })

        add("当前参数", n=c.rf_n_estimators, depth=c.rf_max_depth, leaf=c.rf_min_samples_leaf)
        add("稳健浅树-高叶子", n=min(c.rf_n_estimators, 500), depth=8, leaf=max(c.rf_min_samples_leaf, 30), penalty=max(c.high_volume_down_penalty, 0.12))
        add("启动敏感-低位加权", n=min(c.rf_n_estimators, 500), depth=12, leaf=max(8, min(c.rf_min_samples_leaf, 15)), bonus=max(c.low_volume_up_bonus, 0.08), penalty=c.high_volume_down_penalty)
        add("风险保守-高位惩罚", n=min(c.rf_n_estimators, 500), depth=10, leaf=max(c.rf_min_samples_leaf, 20), bonus=min(c.low_volume_up_bonus, 0.05), penalty=max(c.high_volume_down_penalty, 0.16))
        add("涨停优先", n=min(c.rf_n_estimators, 500), depth=c.rf_max_depth, leaf=c.rf_min_samples_leaf, w_limit=0.78, w_big=0.22)
        add("大涨兼顾", n=min(c.rf_n_estimators, 500), depth=c.rf_max_depth, leaf=c.rf_min_samples_leaf, w_limit=0.55, w_big=0.45)
        add("ExtraTrees-非线性", tree_model="ExtraTrees", n=min(c.rf_n_estimators, 500), depth=c.rf_max_depth, leaf=c.rf_min_samples_leaf)
        add("窄候选高精度", n=min(c.rf_n_estimators, 500), depth=c.rf_max_depth, leaf=max(10, c.rf_min_samples_leaf), topn=max(100, min(c.first_stage_topn, 500)))
        add("宽候选召回", n=min(c.rf_n_estimators, 500), depth=c.rf_max_depth, leaf=max(15, c.rf_min_samples_leaf), topn=max(c.first_stage_topn, 1200))

        # 去重、限量。搜索阶段限制树数量和样本量，最后预测会用被选中的参数重新训练。
        seen = set()
        out = []
        for spec in raw_specs:
            key = tuple((k, spec[k]) for k in sorted(spec) if k != "candidate_name")
            if key in seen:
                continue
            seen.add(key)
            spec["rf_n_estimators_search"] = int(max(80, min(int(spec["rf_n_estimators"]), self.cfg.param_search_max_trees)))
            spec["max_train_rows_search"] = int(max(5000, min(self.cfg.max_train_rows, self.cfg.param_search_max_train_rows)))
            out.append(spec)
            if len(out) >= max(1, int(self.cfg.param_search_candidates)):
                break
        return out

    def _candidate_to_cfg(self, spec: Dict[str, object], *, search_mode: bool) -> AppConfig:
        """把候选参数转成 AppConfig。search_mode=True 时使用快速RF代理评分。"""
        if search_mode:
            return replace(
                self.cfg,
                pathway="RF_ONLY",
                tree_model=str(spec["tree_model"]),
                rf_n_estimators=int(spec["rf_n_estimators_search"]),
                rf_max_depth=int(spec["rf_max_depth"]),
                rf_min_samples_leaf=int(spec["rf_min_samples_leaf"]),
                first_stage_topn=int(spec["first_stage_topn"]),
                max_train_rows=int(spec["max_train_rows_search"]),
                weight_limit=float(spec["weight_limit"]),
                weight_bigrise=float(spec["weight_bigrise"]),
                low_volume_up_bonus=float(spec["low_volume_up_bonus"]),
                high_volume_down_penalty=float(spec["high_volume_down_penalty"]),
            )
        return replace(
            self.cfg,
            tree_model=str(spec["tree_model"]),
            rf_n_estimators=int(spec["rf_n_estimators"]),
            rf_max_depth=int(spec["rf_max_depth"]),
            rf_min_samples_leaf=int(spec["rf_min_samples_leaf"]),
            first_stage_topn=int(spec["first_stage_topn"]),
            weight_limit=float(spec["weight_limit"]),
            weight_bigrise=float(spec["weight_bigrise"]),
            low_volume_up_bonus=float(spec["low_volume_up_bonus"]),
            high_volume_down_penalty=float(spec["high_volume_down_penalty"]),
        )

    def _score_pool_with_probabilities(
        self,
        pool: pd.DataFrame,
        cfg: AppConfig,
        p_limit: np.ndarray,
        p_big: np.ndarray,
        anchor_date: str,
        candidate_name: str,
    ) -> pd.DataFrame:
        out = pool[["ts_code", "trade_date", "close", "pct_chg", "amount", "y_limit", "y_bigrise", "next_pct_chg"]].copy()
        for c in ["name", "industry", "market", "area", "low_volume_up_signal", "high_volume_down_signal",
                  "volume_price_pattern_score", "price_position_20", "price_position_60", "vol_ratio_20", "vol_ratio_60",
                  "amount_ratio_20", "amount_ratio_60"]:
            out[c] = pool[c].values if c in pool.columns else np.nan
        out["candidate_name"] = candidate_name
        out["anchor_trade_date"] = str(anchor_date)
        out["rf_limit_prob"] = p_limit.astype(float)
        out["rf_bigrise_prob"] = p_big.astype(float)
        out["base_score"] = cfg.weight_limit * out["rf_limit_prob"] + cfg.weight_bigrise * out["rf_bigrise_prob"]
        if cfg.use_vp_pattern:
            low_sig = pd.to_numeric(out["low_volume_up_signal"], errors="coerce").fillna(0).clip(0, 3)
            high_risk = pd.to_numeric(out["high_volume_down_signal"], errors="coerce").fillna(0).clip(0, 3)
            out["vp_adjustment"] = cfg.low_volume_up_bonus * low_sig - cfg.high_volume_down_penalty * high_risk
        else:
            out["vp_adjustment"] = 0.0
        out["final_score"] = out["base_score"] + out["vp_adjustment"]
        out["price_band"] = np.where(pd.to_numeric(out["close"], errors="coerce") <= cfg.low_price_cutoff, "低价股", "高价股")
        return out.sort_values("final_score", ascending=False).reset_index(drop=True)

    def _metrics_for_ranked_pool(self, ranked: pd.DataFrame, topn: int) -> Dict[str, float]:
        if ranked.empty:
            return {"day_score": 0.0}
        top = ranked.head(max(1, int(topn))).copy()
        y_limit = pd.to_numeric(ranked["y_limit"], errors="coerce").fillna(0).astype(int).to_numpy()
        y_big = pd.to_numeric(ranked["y_bigrise"], errors="coerce").fillna(0).astype(int).to_numpy()
        p_limit = pd.to_numeric(ranked["rf_limit_prob"], errors="coerce").fillna(0).to_numpy()
        p_big = pd.to_numeric(ranked["rf_bigrise_prob"], errors="coerce").fillna(0).to_numpy()

        top_limit = pd.to_numeric(top["y_limit"], errors="coerce").fillna(0).astype(float)
        top_big = pd.to_numeric(top["y_bigrise"], errors="coerce").fillna(0).astype(float)
        base_limit = float(np.mean(y_limit)) if len(y_limit) else 0.0
        base_big = float(np.mean(y_big)) if len(y_big) else 0.0
        prec_limit = float(top_limit.mean()) if len(top_limit) else 0.0
        prec_big = float(top_big.mean()) if len(top_big) else 0.0
        hit_limit = int(top_limit.sum())
        hit_big = int(top_big.sum())

        def safe_ap(y, p):
            try:
                if len(np.unique(y)) > 1:
                    return float(average_precision_score(y, p))
            except Exception:
                pass
            return 0.0

        ap_limit = safe_ap(y_limit, p_limit)
        ap_big = safe_ap(y_big, p_big)
        lift_limit = prec_limit / max(base_limit, 1e-6)
        lift_big = prec_big / max(base_big, 1e-6)

        # 可靠度评分：既看Top命中，又看相对于全市场基准概率的提升，并加入AP作为排序质量。
        # 涨停更稀缺，权重略高；大涨负责提高稳定性。最后会在3天上求均值并惩罚波动。
        day_score = (
            0.35 * (min(lift_limit, 12.0) / 12.0) +
            0.20 * (min(lift_big, 8.0) / 8.0) +
            0.15 * min(prec_limit * 8.0, 1.0) +
            0.10 * min(prec_big * 4.0, 1.0) +
            0.10 * min(ap_limit * 8.0, 1.0) +
            0.10 * min(ap_big * 4.0, 1.0)
        )
        return {
            "pool_n": int(len(ranked)),
            "topn": int(len(top)),
            "base_limit_rate": base_limit,
            "base_bigrise_rate": base_big,
            "precision_limit_topn": prec_limit,
            "precision_bigrise_topn": prec_big,
            "hit_limit_topn": hit_limit,
            "hit_bigrise_topn": hit_big,
            "lift_limit_topn": float(lift_limit),
            "lift_bigrise_topn": float(lift_big),
            "ap_limit": ap_limit,
            "ap_bigrise": ap_big,
            "day_score": float(day_score),
        }

    def _fit_rf_pair_and_score_anchor(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        cfg: AppConfig,
        anchor_date: str,
        candidate_name: str,
    ) -> Tuple[pd.DataFrame, Dict[str, float]]:
        train = df[(df["trade_date"].astype(str) < str(anchor_date)) & df["y_limit"].notna() & df["y_bigrise"].notna()].copy()
        pool = df[(df["trade_date"].astype(str) == str(anchor_date)) & df["y_limit"].notna() & df["y_bigrise"].notna()].copy()
        if train.empty or pool.empty:
            raise RuntimeError(f"回测日 {anchor_date} 训练集或预测池为空。")
        if train["trade_date"].nunique() < max(20, int(cfg.param_search_min_history_days)):
            raise RuntimeError(f"回测日 {anchor_date} 历史天数不足：{train['trade_date'].nunique()}。")

        rf_limit = TabularBinaryModel(cfg, "y_limit", self.log).fit(train, feature_cols)
        rf_big = TabularBinaryModel(cfg, "y_bigrise", self.log).fit(train, feature_cols)
        p_limit = rf_limit.predict_proba(pool)
        p_big = rf_big.predict_proba(pool)
        ranked = self._score_pool_with_probabilities(pool, cfg, p_limit, p_big, anchor_date, candidate_name)
        metrics = self._metrics_for_ranked_pool(ranked, cfg.param_search_topn)
        return ranked, metrics

    def _auto_select_params_three_day_backtest(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        latest_trade_date: str,
    ) -> AppConfig:
        """最近3个已知交易日滚动预测评分，选择可靠度最高的参数。"""
        all_dates = sorted(df[df["trade_date"].astype(str) < str(latest_trade_date)]["trade_date"].astype(str).unique().tolist())
        if len(all_dates) < max(5, self.cfg.param_search_days):
            self.log("三日自动选参跳过：可回测交易日过少。")
            return self.cfg
        eval_dates = all_dates[-max(1, int(self.cfg.param_search_days)):]
        candidates = self._make_param_candidates()
        self.log("=" * 70)
        self.log("启动三日滚动回测自动选参：用最近已知交易日预测其下一日涨停/大涨，再按可靠度打分。")
        self.log(f"回测交易日: {', '.join(eval_dates)}；候选参数组: {len(candidates)}；每组TopN={self.cfg.param_search_topn}。")
        self.log("说明：选参阶段使用RF/ExtraTrees快速代理评分；最终预测仍按窗口中选择的路径执行，例如 RF_TO_LNN。")

        score_rows = []
        best_candidate_predictions = []
        candidate_ranked_cache: Dict[str, List[pd.DataFrame]] = {}

        for ci, spec in enumerate(candidates, start=1):
            self._check_stop()
            progress_value = 54 + int(18 * (ci - 1) / max(len(candidates), 1))
            self.progress(progress_value, f"三日选参 {ci}/{len(candidates)}")
            search_cfg = self._candidate_to_cfg(spec, search_mode=True)
            self.log(f"[选参 {ci}/{len(candidates)}] {spec['candidate_name']} | tree={search_cfg.tree_model}, trees={search_cfg.rf_n_estimators}, depth={search_cfg.rf_max_depth}, leaf={search_cfg.rf_min_samples_leaf}, bonus={search_cfg.low_volume_up_bonus}, penalty={search_cfg.high_volume_down_penalty}")
            day_scores = []
            day_rows = []
            ranked_list = []
            failed = False
            for d in eval_dates:
                try:
                    ranked, m = self._fit_rf_pair_and_score_anchor(df, feature_cols, search_cfg, d, str(spec["candidate_name"]))
                    ranked_list.append(ranked.head(max(1, int(self.cfg.param_search_topn))).copy())
                    m_row = {"candidate_index": ci, "candidate_name": spec["candidate_name"], "anchor_trade_date": d}
                    m_row.update({k: v for k, v in spec.items() if k not in ["candidate_name"]})
                    m_row.update(m)
                    day_rows.append(m_row)
                    day_scores.append(float(m.get("day_score", 0.0)))
                    self.log(f"    {d}: score={m.get('day_score',0):.4f}, limit_hit={m.get('hit_limit_topn',0)}, bigrise_hit={m.get('hit_bigrise_topn',0)}, lift_limit={m.get('lift_limit_topn',0):.2f}, lift_big={m.get('lift_bigrise_topn',0):.2f}")
                except Exception as e:
                    failed = True
                    self.log(f"    {d}: 跳过，原因：{e}")
                    break
            if failed or not day_scores:
                row = {"candidate_index": ci, "candidate_name": spec["candidate_name"], "aggregate_score": -1.0, "failed": 1}
                row.update({k: v for k, v in spec.items() if k not in ["candidate_name"]})
                score_rows.append(row)
                continue

            mean_score = float(np.mean(day_scores))
            std_score = float(np.std(day_scores))
            aggregate_score = mean_score - 0.20 * std_score
            row = {
                "candidate_index": ci,
                "candidate_name": spec["candidate_name"],
                "aggregate_score": aggregate_score,
                "mean_day_score": mean_score,
                "std_day_score": std_score,
                "failed": 0,
            }
            row.update({k: v for k, v in spec.items() if k not in ["candidate_name"]})
            # 汇总核心指标均值
            for key in ["precision_limit_topn", "precision_bigrise_topn", "hit_limit_topn", "hit_bigrise_topn", "lift_limit_topn", "lift_bigrise_topn", "ap_limit", "ap_bigrise"]:
                vals = [float(x.get(key, 0.0)) for x in day_rows]
                row[f"mean_{key}"] = float(np.mean(vals)) if vals else 0.0
            score_rows.append(row)
            candidate_ranked_cache[str(spec["candidate_name"])] = ranked_list

        scores = pd.DataFrame(score_rows).sort_values("aggregate_score", ascending=False).reset_index(drop=True)
        self.param_search_scores_df = scores
        if scores.empty or float(scores.iloc[0].get("aggregate_score", -1.0)) < 0:
            self.log("三日自动选参没有得到有效候选，继续使用当前窗口参数。")
            return self.cfg

        best = scores.iloc[0].to_dict()
        best_name = str(best.get("candidate_name"))
        self.log("=" * 70)
        self.log(f"三日自动选参完成：最佳参数组 = {best_name}；aggregate_score={float(best.get('aggregate_score',0)):.4f}")
        self.log(f"平均涨停TopN命中={float(best.get('mean_hit_limit_topn',0)):.2f}，平均大涨TopN命中={float(best.get('mean_hit_bigrise_topn',0)):.2f}，平均涨停lift={float(best.get('mean_lift_limit_topn',0)):.2f}。")

        # 保存最佳参数在三个回测日形成的预测名单，便于人工复盘。
        best_preds = candidate_ranked_cache.get(best_name, [])
        if best_preds:
            pred_df = pd.concat(best_preds, axis=0, ignore_index=True)
            pred_df.insert(0, "selected_by_auto_param_search", 1)
            self.param_search_predictions_df = pred_df

        # 把评分最高的参数应用到最终预测。
        best_spec = None
        for spec in candidates:
            if str(spec["candidate_name"]) == best_name:
                best_spec = spec
                break
        if best_spec is None:
            return self.cfg
        final_cfg = self._candidate_to_cfg(best_spec, search_mode=False)
        self.selected_param_summary = {
            "selected_candidate_name": best_name,
            "aggregate_score": float(best.get("aggregate_score", 0.0)),
            "mean_day_score": float(best.get("mean_day_score", 0.0)),
            "std_day_score": float(best.get("std_day_score", 0.0)),
        }
        return final_cfg

    # ------------------------------------------------------------------
    # 对抗验证/市场漂移：不是生成式对抗网络，而是用分布偏移校准风险
    # ------------------------------------------------------------------
    def _compute_adversarial_drift(
        self,
        df_fit: pd.DataFrame,
        latest_df: pd.DataFrame,
        feature_cols: List[str],
    ) -> Tuple[pd.DataFrame, Dict[str, object]]:
        if not bool(getattr(self.cfg, "use_adversarial_validation", True)) or df_fit.empty or latest_df.empty:
            empty = pd.DataFrame(index=latest_df.index)
            empty["adversarial_drift_score"] = 0.0
            empty["drift_penalty"] = 0.0
            return empty, {"enabled": 0, "market_drift_score": 0.0, "top_shift_features": ""}

        numeric_cols = [c for c in feature_cols if c in df_fit.columns and c in latest_df.columns]
        preferred = [
            "pct_chg", "ret1", "ret_3", "ret_5", "ret_10", "ret_20", "ma_gap_5", "ma_gap_10", "ma_gap_20",
            "vol_ratio_5", "vol_ratio_10", "vol_ratio_20", "amount_ratio_5", "amount_ratio_20",
            "volatility_10", "volatility_20", "rsi14", "macd_hist", "net_mf_amount_ratio",
            "large_order_net_ratio", "low_volume_up_signal", "high_volume_down_signal",
            "price_position_20", "price_position_60",
        ]
        use_cols = [c for c in preferred if c in numeric_cols]
        if len(use_cols) < 8:
            use_cols = numeric_cols[:40]
        if not use_cols:
            empty = pd.DataFrame(index=latest_df.index)
            empty["adversarial_drift_score"] = 0.0
            empty["drift_penalty"] = 0.0
            return empty, {"enabled": 0, "market_drift_score": 0.0, "top_shift_features": ""}

        recent_dates = sorted(df_fit["trade_date"].astype(str).unique().tolist())[-80:]
        base = df_fit[df_fit["trade_date"].astype(str).isin(recent_dates)][use_cols].replace([np.inf, -np.inf], np.nan)
        cur = latest_df[use_cols].replace([np.inf, -np.inf], np.nan)
        base_mean = base.mean(axis=0, skipna=True)
        base_std = base.std(axis=0, skipna=True).replace(0, np.nan)
        cur_mean = cur.mean(axis=0, skipna=True)
        shift = ((cur_mean - base_mean).abs() / base_std).replace([np.inf, -np.inf], np.nan).fillna(0)
        row_z = ((cur - base_mean) / base_std).replace([np.inf, -np.inf], np.nan).abs().fillna(0)
        drift_score = row_z.clip(0, 8).mean(axis=1)
        market_drift = float(np.nanmean(shift.clip(0, 8).to_numpy())) if len(shift) else 0.0
        penalty_strength = float(getattr(self.cfg, "drift_penalty_strength", 0.035))
        penalty = np.clip((drift_score - 1.25) * penalty_strength, 0.0, 0.16)
        top_shift = shift.sort_values(ascending=False).head(8)
        out = pd.DataFrame(index=latest_df.index)
        out["adversarial_drift_score"] = drift_score.astype(float)
        out["drift_penalty"] = penalty.astype(float)
        summary = {
            "enabled": 1,
            "market_drift_score": market_drift,
            "top_shift_features": ", ".join([f"{k}:{v:.2f}" for k, v in top_shift.items()]),
        }
        self.log(
            f"对抗验证/漂移检测：market_drift={market_drift:.3f}；"
            f"主要偏移特征：{summary['top_shift_features'] or '无'}。"
        )
        return out, summary

    def _direction_label(self, score: float, risk: float) -> str:
        if risk >= 0.60:
            return "偏弱/高风险"
        if score >= 0.68:
            return "强势看多"
        if score >= 0.56:
            return "偏强"
        if score >= 0.45:
            return "震荡"
        return "偏弱"

    def _reason_text(self, row: pd.Series) -> str:
        reasons = []
        if float(row.get("avg_limit_prob", 0)) >= 0.08:
            reasons.append("涨停概率均值较高")
        if float(row.get("avg_bigrise_prob", 0)) >= 0.18:
            reasons.append("大涨概率均值较高")
        if float(row.get("breadth_up_pct", 0)) >= 0.55:
            reasons.append("上涨家数占优")
        if float(row.get("avg_low_signal", 0)) > float(row.get("avg_high_risk", 0)):
            reasons.append("低位缩量/潜伏信号占优")
        if float(row.get("avg_high_risk", 0)) >= 0.25:
            reasons.append("高位放量下跌风险偏高")
        if float(row.get("avg_drift", 0)) >= 1.8:
            reasons.append("近期分布漂移较大")
        return "；".join(reasons[:4]) if reasons else "模型信号中性，等待确认"

    def _aggregate_view(self, result: pd.DataFrame, group_col: Optional[str], label: str, next_trade_date: str) -> pd.DataFrame:
        df = result.copy()
        if df.empty:
            return pd.DataFrame()
        df["model_score"] = pd.to_numeric(df["final_score"], errors="coerce").fillna(0).clip(lower=0)
        df["limit_consensus"] = 0.5 * pd.to_numeric(df["rf_limit_prob"], errors="coerce").fillna(0) + 0.5 * pd.to_numeric(df["lnn_limit_prob"], errors="coerce").fillna(0)
        df["big_consensus"] = 0.5 * pd.to_numeric(df["rf_bigrise_prob"], errors="coerce").fillna(0) + 0.5 * pd.to_numeric(df["lnn_bigrise_prob"], errors="coerce").fillna(0)
        df["pct_chg_num"] = pd.to_numeric(df["pct_chg"], errors="coerce").fillna(0)
        df["amount_num"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0)
        if group_col is None:
            groups = [("全市场", df)]
        else:
            if group_col not in df.columns:
                return pd.DataFrame()
            groups = [(str(k), v) for k, v in df.groupby(group_col, dropna=False)]
        rows = []
        for name, sub in groups:
            n = len(sub)
            if n <= 0:
                continue
            topn = max(5, min(int(getattr(self.cfg, "sector_topn", 60)), n))
            top = sub.sort_values("model_score", ascending=False).head(topn)
            avg_limit = float(sub["limit_consensus"].mean())
            avg_big = float(sub["big_consensus"].mean())
            avg_score = float(top["model_score"].mean())
            avg_low = float(pd.to_numeric(sub.get("low_volume_up_signal", 0), errors="coerce").fillna(0).mean())
            avg_risk = float(pd.to_numeric(sub.get("high_volume_down_signal", 0), errors="coerce").fillna(0).mean())
            avg_drift = float(pd.to_numeric(sub.get("adversarial_drift_score", 0), errors="coerce").fillna(0).mean())
            breadth = float((sub["pct_chg_num"] > 0).mean())
            risk_score = float(np.clip(0.55 * avg_risk + 0.20 * max(avg_drift - 1.2, 0) + 0.25 * max(0.5 - breadth, 0), 0, 1))
            composite = float(np.clip(0.45 * avg_score + 0.30 * avg_big + 0.15 * avg_limit + 0.10 * breadth - 0.18 * risk_score, 0, 1))
            row = {
                "view_type": label,
                "group_name": name if name and name != "nan" else "未知",
                "predict_for": next_trade_date,
                "stock_count": int(n),
                "topn_used": int(topn),
                "direction": self._direction_label(composite, risk_score),
                "composite_score": composite,
                "risk_score": risk_score,
                "avg_limit_prob": avg_limit,
                "avg_bigrise_prob": avg_big,
                "avg_model_score_topn": avg_score,
                "breadth_up_pct": breadth,
                "avg_pct_chg_today": float(sub["pct_chg_num"].mean()),
                "amount_sum": float(sub["amount_num"].sum()),
                "avg_low_signal": avg_low,
                "avg_high_risk": avg_risk,
                "avg_drift": avg_drift,
                "top_codes": ", ".join(top["ts_code"].astype(str).head(12).tolist()),
            }
            row["conclusion"] = f"{row['group_name']}：{row['direction']}，综合分 {composite:.3f}，风险 {risk_score:.3f}"
            row["reason"] = self._reason_text(pd.Series(row))
            rows.append(row)
        out = pd.DataFrame(rows)
        if out.empty:
            return out
        return out.sort_values(["composite_score", "stock_count"], ascending=[False, False]).reset_index(drop=True)

    def _build_market_sector_reports(self, result: pd.DataFrame, next_trade_date: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if not bool(getattr(self.cfg, "use_market_sector_branch", True)):
            return pd.DataFrame(), pd.DataFrame()
        market_parts = [
            self._aggregate_view(result, None, "全市场", next_trade_date),
            self._aggregate_view(result, "market", "市场板块", next_trade_date),
            self._aggregate_view(result, "price_band", "价格带", next_trade_date),
        ]
        market_summary = pd.concat([x for x in market_parts if x is not None and not x.empty], axis=0, ignore_index=True)
        sector_summary = self._aggregate_view(result, "industry", "行业板块", next_trade_date)
        if not market_summary.empty:
            self.log("大盘/市场预测：" + str(market_summary.iloc[0].get("conclusion", "")) + "；理由：" + str(market_summary.iloc[0].get("reason", "")))
        if not sector_summary.empty:
            leaders = " / ".join(sector_summary.head(5)["group_name"].astype(str).tolist())
            self.log(f"行业强弱预测：前排行业为 {leaders}。")
        return market_summary, sector_summary

    def run(self) -> Dict[str, object]:
        t0 = time.time()
        Path(self.cfg.output_dir).mkdir(parents=True, exist_ok=True)
        self.log("开始构建特征。")
        builder = FeatureBuilder(self.cfg, self.log, self.progress, self.stop_event)
        raw = builder.load_and_merge()
        df, feature_cols, latest_trade_date, next_trade_date = builder.build_features(raw)
        self._check_stop()

        if self.cfg.auto_param_search:
            self.progress(54, "三日回测自动选参")
            selected_cfg = self._auto_select_params_three_day_backtest(df, feature_cols, latest_trade_date)
            self.cfg = selected_cfg
            self.log("最终预测将使用自动选出的参数：" + json.dumps({
                "tree_model": self.cfg.tree_model,
                "tree_backend": self.cfg.tree_backend,
                "rf_n_estimators": self.cfg.rf_n_estimators,
                "rf_max_depth": self.cfg.rf_max_depth,
                "rf_min_samples_leaf": self.cfg.rf_min_samples_leaf,
                "first_stage_topn": self.cfg.first_stage_topn,
                "weight_limit": self.cfg.weight_limit,
                "weight_bigrise": self.cfg.weight_bigrise,
                "low_volume_up_bonus": self.cfg.low_volume_up_bonus,
                "high_volume_down_penalty": self.cfg.high_volume_down_penalty,
            }, ensure_ascii=False))
            self._check_stop()

        latest_df = df[df["trade_date"].astype(str) == str(latest_trade_date)].copy()
        train_df = df[(df["trade_date"].astype(str) < str(latest_trade_date)) & df["y_limit"].notna() & df["y_bigrise"].notna()].copy()
        if len(latest_df) == 0:
            raise RuntimeError("没有最新交易日的预测样本。")
        if len(train_df) < 1000:
            raise RuntimeError(f"训练样本过少：{len(train_df)}。请提前 start_date 或检查数据库。")

        # 按交易日做验证集
        dates = sorted(train_df["trade_date"].astype(str).unique().tolist())
        val_n = max(1, int(len(dates) * self.cfg.validation_days_ratio))
        val_dates = set(dates[-val_n:])
        df_fit = train_df[~train_df["trade_date"].astype(str).isin(val_dates)].copy()
        df_val = train_df[train_df["trade_date"].astype(str).isin(val_dates)].copy()
        if len(df_fit) < 1000:
            df_fit = train_df
            df_val = train_df.iloc[0:0].copy()

        self.log(f"训练集: {len(df_fit):,} 行；验证集: {len(df_val):,} 行；最新预测池: {len(latest_df):,} 只。")
        self.log(f"路径: {self.cfg.pathway}；树模型: {self.cfg.tree_model}；树后端: {self.cfg.tree_backend}。")
        drift_df, drift_summary = self._compute_adversarial_drift(df_fit, latest_df, feature_cols)

        need_rf = self.cfg.pathway in ["RF_TO_LNN", "LNN_TO_RF", "RF_ONLY", "ENSEMBLE"]
        need_lnn = self.cfg.pathway in ["RF_TO_LNN", "LNN_TO_RF", "LNN_ONLY", "ENSEMBLE"]
        need_transformer = bool(getattr(self.cfg, "use_transformer_branch", True))

        rf_limit_prob = np.zeros(len(latest_df), dtype=float)
        rf_big_prob = np.zeros(len(latest_df), dtype=float)
        lnn_limit_prob = np.zeros(len(latest_df), dtype=float)
        lnn_big_prob = np.zeros(len(latest_df), dtype=float)
        transformer_limit_prob = np.zeros(len(latest_df), dtype=float)
        transformer_big_prob = np.zeros(len(latest_df), dtype=float)

        # RF/ExtraTrees
        if need_rf:
            self.progress(60, "训练随机森林/树模型")
            rf_limit = TabularBinaryModel(self.cfg, "y_limit", self.log).fit(df_fit, feature_cols)
            rf_big = TabularBinaryModel(self.cfg, "y_bigrise", self.log).fit(df_fit, feature_cols)
            if len(df_val) > 0:
                try:
                    m1 = rf_limit.validate(df_val)
                    m2 = rf_big.validate(df_val)
                    self.log(f"验证 RF y_limit: {m1}")
                    self.log(f"验证 RF y_bigrise: {m2}")
                except Exception as e:
                    self.log(f"验证 RF 时跳过：{e}")
            rf_limit_prob = rf_limit.predict_proba(latest_df)
            rf_big_prob = rf_big.predict_proba(latest_df)

        self._check_stop()

        # LNN
        if need_lnn:
            self.progress(72, "训练液态神经网络 LNN")
            lnn_limit = LNNBinaryModel(self.cfg, "y_limit", feature_cols, self.log, self.stop_event).fit(df_fit)
            lnn_big = LNNBinaryModel(self.cfg, "y_bigrise", feature_cols, self.log, self.stop_event).fit(df_fit)
            code_to_limit = lnn_limit.predict_latest(df, latest_trade_date)
            code_to_big = lnn_big.predict_latest(df, latest_trade_date)
            latest_codes = latest_df["ts_code"].astype(str).tolist()
            lnn_limit_prob = np.array([code_to_limit.get(c, np.nan) for c in latest_codes], dtype=float)
            lnn_big_prob = np.array([code_to_big.get(c, np.nan) for c in latest_codes], dtype=float)
            # 没有足够序列的股票，用 RF 或 0 填补
            if need_rf:
                lnn_limit_prob = np.where(np.isfinite(lnn_limit_prob), lnn_limit_prob, rf_limit_prob)
                lnn_big_prob = np.where(np.isfinite(lnn_big_prob), lnn_big_prob, rf_big_prob)
            else:
                lnn_limit_prob = np.where(np.isfinite(lnn_limit_prob), lnn_limit_prob, 0.0)
                lnn_big_prob = np.where(np.isfinite(lnn_big_prob), lnn_big_prob, 0.0)

        # Transformer
        if need_transformer:
            self.progress(80, "训练 Transformer 时间序列分支")
            transformer_limit = TransformerBinaryModel(self.cfg, "y_limit", feature_cols, self.log, self.stop_event).fit(df_fit)
            transformer_big = TransformerBinaryModel(self.cfg, "y_bigrise", feature_cols, self.log, self.stop_event).fit(df_fit)
            code_to_limit = transformer_limit.predict_latest(df, latest_trade_date)
            code_to_big = transformer_big.predict_latest(df, latest_trade_date)
            latest_codes = latest_df["ts_code"].astype(str).tolist()
            transformer_limit_prob = np.array([code_to_limit.get(c, np.nan) for c in latest_codes], dtype=float)
            transformer_big_prob = np.array([code_to_big.get(c, np.nan) for c in latest_codes], dtype=float)
            fallback_limit = lnn_limit_prob if need_lnn else rf_limit_prob
            fallback_big = lnn_big_prob if need_lnn else rf_big_prob
            transformer_limit_prob = np.where(np.isfinite(transformer_limit_prob), transformer_limit_prob, fallback_limit)
            transformer_big_prob = np.where(np.isfinite(transformer_big_prob), transformer_big_prob, fallback_big)

        self.progress(88, "融合评分并生成候选")
        result = latest_df[["ts_code", "trade_date", "close", "pct_chg", "amount"]].copy()
        for c in ["name", "industry", "market", "area"]:
            result[c] = latest_df[c].values if c in latest_df.columns else ""
        for c in [
            "low_volume_up_signal", "high_volume_down_signal", "volume_price_pattern_score",
            "price_position_20", "price_position_60", "vol_ratio_20", "vol_ratio_60",
            "amount_ratio_20", "amount_ratio_60", "low_volume_up_flag_20",
            "low_volume_up_flag_60", "high_volume_down_flag_20", "high_volume_down_flag_60"
        ]:
            result[c] = latest_df[c].values if c in latest_df.columns else np.nan
        result["adversarial_drift_score"] = drift_df.reindex(latest_df.index)["adversarial_drift_score"].to_numpy() if "adversarial_drift_score" in drift_df.columns else 0.0
        result["drift_penalty"] = drift_df.reindex(latest_df.index)["drift_penalty"].to_numpy() if "drift_penalty" in drift_df.columns else 0.0
        result["rf_limit_prob"] = rf_limit_prob
        result["rf_bigrise_prob"] = rf_big_prob
        result["lnn_limit_prob"] = lnn_limit_prob
        result["lnn_bigrise_prob"] = lnn_big_prob
        result["transformer_limit_prob"] = transformer_limit_prob
        result["transformer_bigrise_prob"] = transformer_big_prob
        result["rf_score"] = self.cfg.weight_limit * result["rf_limit_prob"] + self.cfg.weight_bigrise * result["rf_bigrise_prob"]
        result["lnn_score"] = self.cfg.weight_limit * result["lnn_limit_prob"] + self.cfg.weight_bigrise * result["lnn_bigrise_prob"]
        result["transformer_score"] = self.cfg.weight_limit * result["transformer_limit_prob"] + self.cfg.weight_bigrise * result["transformer_bigrise_prob"]
        fw_tree = max(float(getattr(self.cfg, "fusion_weight_tree", 0.40)), 0.0) if need_rf else 0.0
        fw_lnn = max(float(getattr(self.cfg, "fusion_weight_lnn", 0.30)), 0.0) if need_lnn else 0.0
        fw_tf = max(float(getattr(self.cfg, "fusion_weight_transformer", 0.30)), 0.0) if need_transformer else 0.0
        fw_sum = max(fw_tree + fw_lnn + fw_tf, 1e-9)
        result["fusion_score"] = (
            fw_tree / fw_sum * result["rf_score"] +
            fw_lnn / fw_sum * result["lnn_score"] +
            fw_tf / fw_sum * result["transformer_score"]
        )

        if self.cfg.pathway == "RF_ONLY":
            result["final_score"] = result["rf_score"]
            result["stage_selected"] = True
        elif self.cfg.pathway == "LNN_ONLY":
            result["final_score"] = result["lnn_score"]
            result["stage_selected"] = True
        elif self.cfg.pathway == "ENSEMBLE":
            result["final_score"] = 0.5 * result["rf_score"] + 0.5 * result["lnn_score"]
            result["stage_selected"] = True
        elif self.cfg.pathway == "LNN_TO_RF":
            cutoff_codes = set(result.sort_values("lnn_score", ascending=False).head(self.cfg.first_stage_topn)["ts_code"].astype(str))
            result["stage_selected"] = result["ts_code"].astype(str).isin(cutoff_codes)
            result["final_score"] = np.where(result["stage_selected"], result["rf_score"], -1.0)
        else:  # RF_TO_LNN
            cutoff_codes = set(result.sort_values("rf_score", ascending=False).head(self.cfg.first_stage_topn)["ts_code"].astype(str))
            result["stage_selected"] = result["ts_code"].astype(str).isin(cutoff_codes)
            result["final_score"] = np.where(result["stage_selected"], result["lnn_score"], -1.0)

        if bool(getattr(self.cfg, "use_registry_fusion", True)) and need_transformer:
            selected_codes = set()
            for score_col in ["rf_score", "lnn_score", "transformer_score", "fusion_score"]:
                selected_codes.update(result.sort_values(score_col, ascending=False).head(self.cfg.first_stage_topn)["ts_code"].astype(str).tolist())
            result["stage_selected"] = result["ts_code"].astype(str).isin(selected_codes)
            result["final_score"] = np.where(result["stage_selected"], result["fusion_score"], -1.0)
            self.log(
                f"已启用 XGBoost/RF + LNN + Transformer 融合："
                f"tree={fw_tree/fw_sum:.2f}, lnn={fw_lnn/fw_sum:.2f}, transformer={fw_tf/fw_sum:.2f}"
            )

        result["final_score_raw"] = result["final_score"].astype(float)
        if self.cfg.use_vp_pattern:
            low_sig = pd.to_numeric(result["low_volume_up_signal"], errors="coerce").fillna(0).clip(0, 3)
            high_risk = pd.to_numeric(result["high_volume_down_signal"], errors="coerce").fillna(0).clip(0, 3)
            result["vp_adjustment"] = self.cfg.low_volume_up_bonus * low_sig - self.cfg.high_volume_down_penalty * high_risk
            result["final_score"] = np.where(result["stage_selected"], result["final_score_raw"] + result["vp_adjustment"], -1.0)
            self.log(
                "已启用量价形态修正：低位缩量上涨加分，"
                f"高位放量下跌扣分；bonus={self.cfg.low_volume_up_bonus}, "
                f"penalty={self.cfg.high_volume_down_penalty}。"
            )
        else:
            result["vp_adjustment"] = 0.0
        if bool(getattr(self.cfg, "use_adversarial_validation", True)):
            result["final_score"] = np.where(result["stage_selected"], result["final_score"] - result["drift_penalty"], -1.0)
            self.log("已启用对抗验证漂移扣分：行情特征偏离近期训练分布越大，候选分数越保守。")

        result["price_band"] = np.where(result["close"] <= self.cfg.low_price_cutoff, "低价股", "高价股")
        result["run_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        result["predict_for"] = next_trade_date
        result["pathway"] = self.cfg.pathway
        result["tree_model"] = self.cfg.tree_model

        result = result.sort_values("final_score", ascending=False).reset_index(drop=True)
        low_df = result[(result["price_band"] == "低价股") & (result["stage_selected"] == True)].sort_values("final_score", ascending=False).head(self.cfg.final_topn_each_band)
        high_df = result[(result["price_band"] == "高价股") & (result["stage_selected"] == True)].sort_values("final_score", ascending=False).head(self.cfg.final_topn_each_band)
        all_top = pd.concat([low_df, high_df], axis=0).sort_values("final_score", ascending=False)
        pools = self._build_candidate_pools(result) if self.cfg.use_pool_output else {}
        if pools:
            self.log("已生成分池榜单：强势延续池 / 启动观察池 / 低位潜伏池 / 风险排除池。")
        market_summary, sector_summary = self._build_market_sector_reports(result, next_trade_date)

        self.progress(94, "保存结果")
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_all = os.path.join(self.cfg.output_dir, f"nextday_limit_lnn_rf_all_{run_id}.csv")
        csv_low = os.path.join(self.cfg.output_dir, f"nextday_limit_lnn_rf_low_price_{run_id}.csv")
        csv_high = os.path.join(self.cfg.output_dir, f"nextday_limit_lnn_rf_high_price_{run_id}.csv")
        csv_pools = os.path.join(self.cfg.output_dir, f"nextday_limit_lnn_rf_pools_{run_id}.csv")
        csv_market = os.path.join(self.cfg.output_dir, f"nextday_market_sector_summary_{run_id}.csv")
        csv_sector = os.path.join(self.cfg.output_dir, f"nextday_industry_sector_summary_{run_id}.csv")
        xlsx_path = os.path.join(self.cfg.output_dir, f"nextday_limit_lnn_rf_result_{run_id}.xlsx")

        result.to_csv(csv_all, index=False, encoding="utf-8-sig")
        low_df.to_csv(csv_low, index=False, encoding="utf-8-sig")
        high_df.to_csv(csv_high, index=False, encoding="utf-8-sig")
        if pools:
            pd.concat(list(pools.values()), axis=0, ignore_index=True).to_csv(csv_pools, index=False, encoding="utf-8-sig")
        if not market_summary.empty:
            market_summary.to_csv(csv_market, index=False, encoding="utf-8-sig")
        if not sector_summary.empty:
            sector_summary.to_csv(csv_sector, index=False, encoding="utf-8-sig")
        lmstudio_context = ""
        if bool(getattr(self.cfg, "export_lmstudio_context", True)):
            try:
                lmstudio_context = self._export_lmstudio_context(
                    run_id=run_id,
                    latest_trade_date=latest_trade_date,
                    next_trade_date=next_trade_date,
                    result=result,
                    market_summary=market_summary,
                    sector_summary=sector_summary,
                    drift_summary=drift_summary,
                )
            except Exception as e:
                self.log(f"LM Studio 解读上下文导出失败：{e}")
        try:
            with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
                low_df.to_excel(writer, sheet_name="低价候选", index=False)
                high_df.to_excel(writer, sheet_name="高价候选", index=False)
                all_top.to_excel(writer, sheet_name="综合Top", index=False)
                if pools:
                    pools.get("strong_continuation", pd.DataFrame()).to_excel(writer, sheet_name="强势延续池", index=False)
                    pools.get("startup_watch", pd.DataFrame()).to_excel(writer, sheet_name="启动观察池", index=False)
                    pools.get("low_latent", pd.DataFrame()).to_excel(writer, sheet_name="低位潜伏池", index=False)
                    pools.get("risk_exclusion", pd.DataFrame()).to_excel(writer, sheet_name="风险排除池", index=False)
                result.head(1000).to_excel(writer, sheet_name="全市场Top1000", index=False)
                if self.param_search_scores_df is not None and not self.param_search_scores_df.empty:
                    self.param_search_scores_df.to_excel(writer, sheet_name="三日参数评分", index=False)
                if self.param_search_predictions_df is not None and not self.param_search_predictions_df.empty:
                    self.param_search_predictions_df.head(2000).to_excel(writer, sheet_name="三日回测预测", index=False)
                if not market_summary.empty:
                    market_summary.to_excel(writer, sheet_name="大盘与市场板块", index=False)
                if not sector_summary.empty:
                    sector_summary.head(300).to_excel(writer, sheet_name="行业板块预测", index=False)
                pd.DataFrame([drift_summary]).to_excel(writer, sheet_name="对抗漂移", index=False)
                pd.DataFrame([asdict(self.cfg)]).to_excel(writer, sheet_name="参数", index=False)
        except Exception as e:
            self.log(f"Excel 导出跳过：{e}。可使用 CSV 文件。")
            xlsx_path = ""

        self._write_back_to_db(run_id, latest_trade_date, next_trade_date, result, low_df, high_df, pools, market_summary, sector_summary, drift_summary)
        elapsed = time.time() - t0
        self.progress(100, "完成")
        self.log(f"完成。耗时 {elapsed/60:.2f} 分钟。")
        self.log(f"输出目录: {self.cfg.output_dir}")

        return {
            "run_id": run_id,
            "latest_trade_date": latest_trade_date,
            "predict_for": next_trade_date,
            "csv_all": csv_all,
            "csv_low": csv_low,
            "csv_high": csv_high,
            "csv_pools": csv_pools,
            "csv_market": csv_market,
            "csv_sector": csv_sector,
            "lmstudio_context": lmstudio_context,
            "xlsx": xlsx_path,
            "low_df": low_df,
            "high_df": high_df,
            "all_top": all_top,
            "strong_continuation_df": pools.get("strong_continuation", pd.DataFrame()),
            "startup_watch_df": pools.get("startup_watch", pd.DataFrame()),
            "low_latent_df": pools.get("low_latent", pd.DataFrame()),
            "risk_exclusion_df": pools.get("risk_exclusion", pd.DataFrame()),
            "market_summary_df": market_summary,
            "sector_summary_df": sector_summary,
        }

    def _export_lmstudio_context(
        self,
        run_id: str,
        latest_trade_date: str,
        next_trade_date: str,
        result: pd.DataFrame,
        market_summary: pd.DataFrame,
        sector_summary: pd.DataFrame,
        drift_summary: Dict[str, object],
    ) -> str:
        topn = int(getattr(self.cfg, "lmstudio_context_topn", 40) or 40)
        cols = [
            "ts_code", "name", "industry", "market", "trade_date", "predict_for",
            "final_score", "fusion_score", "rf_score", "lnn_score", "transformer_score",
            "rf_limit_prob", "lnn_limit_prob", "transformer_limit_prob",
            "rf_bigrise_prob", "lnn_bigrise_prob", "transformer_bigrise_prob",
            "low_volume_up_signal", "high_volume_down_signal", "adversarial_drift_score",
            "drift_penalty", "price_band", "stage_selected",
        ]
        use_cols = [c for c in cols if c in result.columns]
        top_rows = result.sort_values("final_score", ascending=False).head(topn)[use_cols].copy()
        context = {
            "instruction": (
                "你是本地 LM Studio 解释层。只解释模型输出，不参与核心打分。"
                "请先给结论，再说明理由、风险、板块共振和需要盘中确认的条件。"
            ),
            "run_id": run_id,
            "latest_trade_date": latest_trade_date,
            "predict_for": next_trade_date,
            "model_policy": {
                "core_scoring": "XGBoost/RF + LNN + Transformer + feedback/drift adjustment",
                "language_model_role": "explanation_only",
                "not_in_core_score": True,
            },
            "config": {
                "pathway": self.cfg.pathway,
                "tree_backend": self.cfg.tree_backend,
                "use_transformer_branch": bool(getattr(self.cfg, "use_transformer_branch", True)),
                "use_model_registry": bool(getattr(self.cfg, "use_model_registry", True)),
                "fusion_weight_tree": float(getattr(self.cfg, "fusion_weight_tree", 0.40)),
                "fusion_weight_lnn": float(getattr(self.cfg, "fusion_weight_lnn", 0.30)),
                "fusion_weight_transformer": float(getattr(self.cfg, "fusion_weight_transformer", 0.30)),
            },
            "top_candidates": top_rows.to_dict(orient="records"),
            "market_summary": market_summary.head(20).to_dict(orient="records") if isinstance(market_summary, pd.DataFrame) and not market_summary.empty else [],
            "sector_summary": sector_summary.head(30).to_dict(orient="records") if isinstance(sector_summary, pd.DataFrame) and not sector_summary.empty else [],
            "drift_summary": drift_summary or {},
        }
        path = os.path.join(self.cfg.output_dir, f"lmstudio_context_{run_id}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(context, f, ensure_ascii=False, indent=2)
        self.log(f"LM Studio 解读上下文已导出：{path}")
        return path

    def _prepare_pool_scores(self, result: pd.DataFrame) -> pd.DataFrame:
        """为分池输出准备更可解释的分数。"""
        out = result.copy()
        for c in ["final_score", "final_score_raw", "rf_limit_prob", "rf_bigrise_prob", "lnn_limit_prob", "lnn_bigrise_prob",
                  "transformer_limit_prob", "transformer_bigrise_prob", "transformer_score", "fusion_score",
                  "low_volume_up_signal", "high_volume_down_signal", "price_position_20", "price_position_60", "pct_chg"]:
            if c not in out.columns:
                out[c] = 0.0
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)

        limit_consensus = (out["rf_limit_prob"] + out["lnn_limit_prob"] + out["transformer_limit_prob"]) / 3.0
        big_consensus = (out["rf_bigrise_prob"] + out["lnn_bigrise_prob"] + out["transformer_bigrise_prob"]) / 3.0
        model_agreement = 1.0 - out[["rf_limit_prob", "lnn_limit_prob", "transformer_limit_prob"]].std(axis=1).fillna(0).clip(0, 1)
        position_mean = out[["price_position_20", "price_position_60"]].mean(axis=1).clip(0, 1).fillna(0)
        low_position_bonus = (1.0 - position_mean).fillna(0)

        out["continuation_score"] = (
            0.50 * out["final_score"] +
            0.25 * limit_consensus +
            0.15 * big_consensus +
            0.10 * model_agreement -
            0.10 * out["high_volume_down_signal"].clip(0, 3)
        )
        out["startup_score"] = (
            0.55 * out["final_score"] +
            0.20 * big_consensus +
            0.15 * out["low_volume_up_signal"].clip(0, 3) +
            0.10 * low_position_bonus -
            0.12 * out["high_volume_down_signal"].clip(0, 3)
        )
        out["latent_score"] = (
            0.40 * out["final_score"] +
            0.25 * out["low_volume_up_signal"].clip(0, 3) +
            0.20 * low_position_bonus +
            0.15 * big_consensus -
            0.15 * out["high_volume_down_signal"].clip(0, 3)
        )
        out["risk_score"] = (
            out["high_volume_down_signal"].clip(0, 3) +
            0.20 * position_mean +
            0.10 * np.maximum(-out["pct_chg"], 0) / 10.0
        )
        return out

    def _build_candidate_pools(self, result: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """把总排名拆成：强势延续、启动观察、低位潜伏、风险排除。"""
        df = self._prepare_pool_scores(result)
        selected = df["stage_selected"].astype(bool) if "stage_selected" in df.columns else pd.Series(True, index=df.index)
        pct = pd.to_numeric(df["pct_chg"], errors="coerce").fillna(0)
        low_sig = pd.to_numeric(df["low_volume_up_signal"], errors="coerce").fillna(0)
        high_risk = pd.to_numeric(df["high_volume_down_signal"], errors="coerce").fillna(0)
        n = max(1, int(getattr(self.cfg, "pool_topn_each", self.cfg.final_topn_each_band)))
        risk_cut = float(getattr(self.cfg, "risk_high_signal_min", 0.25))
        low_cut = float(getattr(self.cfg, "latent_low_signal_min", 0.05))
        safe = selected & (high_risk < risk_cut)

        strong_mask = safe & (pct >= float(self.cfg.limit_threshold))
        startup_mask = safe & (pct >= float(getattr(self.cfg, "startup_min_pct", 2.0))) & (pct < float(self.cfg.limit_threshold))
        latent_mask = safe & (pct >= float(getattr(self.cfg, "latent_min_pct", 0.0))) & (pct <= float(getattr(self.cfg, "latent_max_pct", 5.0))) & (low_sig >= low_cut)
        risk_mask = selected & (high_risk >= risk_cut)

        pools = {
            "strong_continuation": df[strong_mask].sort_values("continuation_score", ascending=False).head(n).copy(),
            "startup_watch": df[startup_mask].sort_values("startup_score", ascending=False).head(n).copy(),
            "low_latent": df[latent_mask].sort_values("latent_score", ascending=False).head(n).copy(),
            "risk_exclusion": df[risk_mask].sort_values("risk_score", ascending=False).head(n).copy(),
        }
        names = {
            "strong_continuation": "强势延续池",
            "startup_watch": "启动观察池",
            "low_latent": "低位潜伏池",
            "risk_exclusion": "风险排除池",
        }
        for k, v in pools.items():
            if v is not None and not v.empty:
                v.insert(0, "pool_name", names[k])
        return pools

    def _write_back_to_db(
        self,
        run_id: str,
        latest_trade_date: str,
        next_trade_date: str,
        result: pd.DataFrame,
        low_df: pd.DataFrame,
        high_df: pd.DataFrame,
        pools: Optional[Dict[str, pd.DataFrame]] = None,
        market_summary: Optional[pd.DataFrame] = None,
        sector_summary: Optional[pd.DataFrame] = None,
        drift_summary: Optional[Dict[str, object]] = None,
    ):
        conn = connect_db(self.cfg.db_path)
        try:
            run_row = pd.DataFrame([{
                "run_id": run_id,
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "latest_trade_date": latest_trade_date,
                "predict_for": next_trade_date,
                "pathway": self.cfg.pathway,
                "tree_model": self.cfg.tree_model,
                "tree_backend": self.cfg.tree_backend,
                "config_json": json.dumps(asdict(self.cfg), ensure_ascii=False),
            }])
            # 兼容旧版本数据库：如果表已存在但缺少新版本新增列，先自动 ALTER TABLE。
            ensure_table_accepts_dataframe(conn, "ml_prediction_runs", run_row, self.log)
            conn.execute("BEGIN;")
            append_df_fast(run_row, "ml_prediction_runs", conn)

            cand = result.copy()
            cand.insert(0, "run_id", run_id)
            ensure_table_accepts_dataframe(conn, "ml_prediction_candidates", cand, self.log)
            append_df_fast(cand, "ml_prediction_candidates", conn)

            if pools:
                pool_frames = []
                for pool_key, pool_df in pools.items():
                    if pool_df is None or pool_df.empty:
                        continue
                    tmp = pool_df.copy()
                    tmp.insert(0, "run_id", run_id)
                    tmp.insert(1, "pool_key", pool_key)
                    pool_frames.append(tmp)
                if pool_frames:
                    pool_all = pd.concat(pool_frames, axis=0, ignore_index=True)
                    ensure_table_accepts_dataframe(conn, "ml_prediction_candidate_pools", pool_all, self.log)
                    append_df_fast(pool_all, "ml_prediction_candidate_pools", conn)

            if self.param_search_scores_df is not None and not self.param_search_scores_df.empty:
                score_df = self.param_search_scores_df.copy()
                score_df.insert(0, "run_id", run_id)
                score_df.insert(1, "created_at", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                ensure_table_accepts_dataframe(conn, "ml_param_search_scores", score_df, self.log)
                append_df_fast(score_df, "ml_param_search_scores", conn)

            if self.param_search_predictions_df is not None and not self.param_search_predictions_df.empty:
                pred_df = self.param_search_predictions_df.copy()
                pred_df.insert(0, "run_id", run_id)
                pred_df.insert(1, "created_at", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                ensure_table_accepts_dataframe(conn, "ml_backtest_predictions_3d", pred_df, self.log)
                append_df_fast(pred_df, "ml_backtest_predictions_3d", conn)

            summary_frames = []
            for summary_name, summary_df in [("market", market_summary), ("sector", sector_summary)]:
                if summary_df is None or summary_df.empty:
                    continue
                tmp = summary_df.copy()
                tmp.insert(0, "run_id", run_id)
                tmp.insert(1, "created_at", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                tmp.insert(2, "summary_name", summary_name)
                summary_frames.append(tmp)
            if summary_frames:
                summary_all = pd.concat(summary_frames, axis=0, ignore_index=True)
                ensure_table_accepts_dataframe(conn, "ml_market_sector_summary", summary_all, self.log)
                append_df_fast(summary_all, "ml_market_sector_summary", conn)

            if drift_summary:
                drift_row = pd.DataFrame([dict(drift_summary)])
                drift_row.insert(0, "run_id", run_id)
                drift_row.insert(1, "created_at", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                drift_row.insert(2, "latest_trade_date", latest_trade_date)
                drift_row.insert(3, "predict_for", next_trade_date)
                ensure_table_accepts_dataframe(conn, "ml_adversarial_drift_summary", drift_row, self.log)
                append_df_fast(drift_row, "ml_adversarial_drift_summary", conn)

            conn.commit()
            self.log("结果已写回数据库表：ml_prediction_runs / ml_prediction_candidates / ml_prediction_candidate_pools / ml_market_sector_summary / ml_adversarial_drift_summary 等。")
        except Exception:
            try:
                conn.rollback()
            except Exception:
                pass
            raise
        finally:
            conn.close()


# -----------------------------
# GUI
# -----------------------------

class StockPredictorGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("下一交易日涨停/大涨预测 - 分池版 RF + LNN 控制台")
        self.geometry("1280x860")
        self.minsize(1120, 760)
        self.log_queue: "queue.Queue" = None  # type: ignore
        import queue
        self.log_queue = queue.Queue()
        self.worker: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        self.current_result: Optional[Dict[str, object]] = None
        self._init_vars()
        self._build_ui()
        self.after(150, self._poll_queue)

    def _init_vars(self):
        default_db = r"data\tu_share_data.db"
        default_out = r"ml_outputs"
        self.db_var = tk.StringVar(value=default_db)
        self.out_var = tk.StringVar(value=default_out)
        self.start_date_var = tk.StringVar(value="20200101")
        self.pathway_var = tk.StringVar(value="RF_TO_LNN")
        self.tree_model_var = tk.StringVar(value="RandomForest")
        self.tree_backend_var = tk.StringVar(value="auto")

        self.exclude_bj_var = tk.BooleanVar(value=True)
        self.exclude_st_var = tk.BooleanVar(value=True)
        self.use_4090_var = tk.BooleanVar(value=True)
        self.use_5090_var = tk.BooleanVar(value=True)
        self.cpu_fallback_var = tk.BooleanVar(value=True)
        self.use_dual_gpu_parallel_var = tk.BooleanVar(value=True)
        self.primary_gpu_var = tk.StringVar(value="0")
        self.secondary_gpu_var = tk.StringVar(value="1")
        self.rf_class_weight_var = tk.BooleanVar(value=True)

        self.min_amount_var = tk.StringVar(value="50000")
        self.min_close_var = tk.StringVar(value="1")
        self.max_close_var = tk.StringVar(value="300")
        self.low_price_cutoff_var = tk.StringVar(value="20")
        self.limit_threshold_var = tk.StringVar(value="9.5")
        self.bigrise_threshold_var = tk.StringVar(value="5")
        self.final_topn_var = tk.StringVar(value="100")
        self.use_pool_output_var = tk.BooleanVar(value=True)
        self.pool_topn_each_var = tk.StringVar(value="30")
        self.max_auto_weight_limit_var = tk.StringVar(value="0.70")
        self.startup_min_pct_var = tk.StringVar(value="2.0")
        self.latent_min_pct_var = tk.StringVar(value="0.0")
        self.latent_max_pct_var = tk.StringVar(value="5.0")
        self.latent_low_signal_min_var = tk.StringVar(value="0.05")
        self.risk_high_signal_min_var = tk.StringVar(value="0.25")
        self.first_stage_topn_var = tk.StringVar(value="800")
        self.max_train_rows_var = tk.StringVar(value="250000")
        self.feature_history_days_var = tk.StringVar(value="900")

        self.rf_n_estimators_var = tk.StringVar(value="600")
        self.rf_max_depth_var = tk.StringVar(value="10")
        self.rf_min_leaf_var = tk.StringVar(value="20")
        self.rf_n_jobs_var = tk.StringVar(value="-1")

        self.lnn_seq_len_var = tk.StringVar(value="20")
        self.lnn_hidden_var = tk.StringVar(value="96")
        self.lnn_epochs_var = tk.StringVar(value="8")
        self.lnn_batch_var = tk.StringVar(value="1024")
        self.lnn_lr_var = tk.StringVar(value="0.001")
        self.lnn_dropout_var = tk.StringVar(value="0.10")

        self.weight_limit_var = tk.StringVar(value="0.65")
        self.weight_bigrise_var = tk.StringVar(value="0.35")
        self.use_vp_pattern_var = tk.BooleanVar(value=True)
        self.vp_low_pos_cutoff_var = tk.StringVar(value="0.35")
        self.vp_high_pos_cutoff_var = tk.StringVar(value="0.70")
        self.vp_shrink_cutoff_var = tk.StringVar(value="0.80")
        self.vp_expand_cutoff_var = tk.StringVar(value="1.50")
        self.low_volume_up_bonus_var = tk.StringVar(value="0.06")
        self.high_volume_down_penalty_var = tk.StringVar(value="0.10")

        self.auto_param_search_var = tk.BooleanVar(value=True)
        self.param_search_days_var = tk.StringVar(value="3")
        self.param_search_topn_var = tk.StringVar(value="80")
        self.param_search_candidates_var = tk.StringVar(value="4")
        self.param_search_max_train_rows_var = tk.StringVar(value="60000")
        self.param_search_max_trees_var = tk.StringVar(value="220")
        self.param_search_min_history_days_var = tk.StringVar(value="120")
        self.use_market_sector_branch_var = tk.BooleanVar(value=True)
        self.use_adversarial_validation_var = tk.BooleanVar(value=True)
        self.drift_penalty_strength_var = tk.StringVar(value="0.035")
        self.sector_topn_var = tk.StringVar(value="60")

        self.external_script_var = tk.StringVar(value="")
        self.progress_var = tk.IntVar(value=0)
        self.status_var = tk.StringVar(value="就绪")

    def _build_ui(self):
        root = ttk.Frame(self, padding=10)
        root.pack(fill=tk.BOTH, expand=True)

        top = ttk.LabelFrame(root, text="数据库与输出")
        top.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(top, text="SQLite数据库：").grid(row=0, column=0, sticky="w", padx=6, pady=6)
        ttk.Entry(top, textvariable=self.db_var, width=92).grid(row=0, column=1, sticky="we", padx=6, pady=6)
        ttk.Button(top, text="选择", command=self._browse_db).grid(row=0, column=2, padx=6)
        ttk.Label(top, text="输出目录：").grid(row=1, column=0, sticky="w", padx=6, pady=6)
        ttk.Entry(top, textvariable=self.out_var, width=92).grid(row=1, column=1, sticky="we", padx=6, pady=6)
        ttk.Button(top, text="选择", command=self._browse_out).grid(row=1, column=2, padx=6)
        top.columnconfigure(1, weight=1)

        mid = ttk.Frame(root)
        mid.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(mid)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 8))
        right = ttk.Frame(mid)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self._build_params(left)
        self._build_logs_and_results(right)

        bottom = ttk.Frame(root)
        bottom.pack(fill=tk.X, pady=(8, 0))
        ttk.Progressbar(bottom, orient="horizontal", mode="determinate", maximum=100, variable=self.progress_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        ttk.Label(bottom, textvariable=self.status_var, width=34).pack(side=tk.LEFT)

    def _build_params(self, parent):
        nb = ttk.Notebook(parent)
        nb.pack(fill=tk.Y, expand=False)

        page1 = ttk.Frame(nb, padding=8)
        page2 = ttk.Frame(nb, padding=8)
        page3 = ttk.Frame(nb, padding=8)
        page4 = ttk.Frame(nb, padding=8)
        page5 = ttk.Frame(nb, padding=8)
        page6 = ttk.Frame(nb, padding=8)
        page7 = ttk.Frame(nb, padding=8)
        nb.add(page1, text="任务")
        nb.add(page2, text="随机森林")
        nb.add(page3, text="LNN/GPU")
        nb.add(page5, text="量价形态")
        nb.add(page6, text="三日选参")
        nb.add(page7, text="分池输出")
        nb.add(page4, text="扩展接口")

        r = 0
        ttk.Label(page1, text="训练起始日期：").grid(row=r, column=0, sticky="w", pady=4)
        ttk.Entry(page1, textvariable=self.start_date_var, width=18).grid(row=r, column=1, sticky="w", pady=4)
        r += 1
        ttk.Label(page1, text="预测路径：").grid(row=r, column=0, sticky="w", pady=4)
        ttk.Combobox(page1, textvariable=self.pathway_var, width=18, state="readonly", values=[
            "RF_TO_LNN", "LNN_TO_RF", "RF_ONLY", "LNN_ONLY", "ENSEMBLE"
        ]).grid(row=r, column=1, sticky="w", pady=4)
        r += 1
        ttk.Label(page1, text="树模型：").grid(row=r, column=0, sticky="w", pady=4)
        ttk.Combobox(page1, textvariable=self.tree_model_var, width=18, state="readonly", values=["RandomForest", "ExtraTrees"]).grid(row=r, column=1, sticky="w", pady=4)
        r += 1
        ttk.Checkbutton(page1, text="排除北交所 .BJ", variable=self.exclude_bj_var).grid(row=r, column=0, columnspan=2, sticky="w", pady=3)
        r += 1
        ttk.Checkbutton(page1, text="排除 ST/退市风险", variable=self.exclude_st_var).grid(row=r, column=0, columnspan=2, sticky="w", pady=3)
        r += 1

        for label, var in [
            ("最小成交额/amount", self.min_amount_var),
            ("最低价格", self.min_close_var),
            ("最高价格", self.max_close_var),
            ("低/高价分界", self.low_price_cutoff_var),
            ("涨停阈值%", self.limit_threshold_var),
            ("大涨阈值%", self.bigrise_threshold_var),
            ("每板块TopN", self.final_topn_var),
            ("第一阶段候选N", self.first_stage_topn_var),
            ("最大训练行数", self.max_train_rows_var),
            ("读取历史交易日数", self.feature_history_days_var),
            ("涨停权重", self.weight_limit_var),
            ("大涨权重", self.weight_bigrise_var),
        ]:
            ttk.Label(page1, text=label + "：").grid(row=r, column=0, sticky="w", pady=4)
            ttk.Entry(page1, textvariable=var, width=18).grid(row=r, column=1, sticky="w", pady=4)
            r += 1

        ttk.Button(page1, text="开始预测", command=self._start).grid(row=r, column=0, sticky="we", pady=(12, 4))
        ttk.Button(page1, text="停止", command=self._stop).grid(row=r, column=1, sticky="we", pady=(12, 4))
        r += 1
        ttk.Button(page1, text="保存参数", command=self._save_config).grid(row=r, column=0, sticky="we", pady=4)
        ttk.Button(page1, text="读取参数", command=self._load_config).grid(row=r, column=1, sticky="we", pady=4)

        r = 0
        ttk.Label(page2, text="树模型后端：").grid(row=r, column=0, sticky="w", pady=5)
        ttk.Combobox(page2, textvariable=self.tree_backend_var, width=18, state="readonly", values=["auto", "cuml", "xgboost_gpu", "sklearn"]).grid(row=r, column=1, sticky="w", pady=5)
        r += 1
        for label, var in [
            ("树数量 n_estimators", self.rf_n_estimators_var),
            ("最大深度 max_depth", self.rf_max_depth_var),
            ("叶子最小样本", self.rf_min_leaf_var),
            ("CPU线程 n_jobs", self.rf_n_jobs_var),
        ]:
            ttk.Label(page2, text=label + "：").grid(row=r, column=0, sticky="w", pady=5)
            ttk.Entry(page2, textvariable=var, width=18).grid(row=r, column=1, sticky="w", pady=5)
            r += 1
        ttk.Checkbutton(page2, text="类别不平衡自动加权", variable=self.rf_class_weight_var).grid(row=r, column=0, columnspan=2, sticky="w", pady=5)
        r += 1
        ttk.Label(page2, text="auto 会依次尝试 cuML GPU、XGBoost GPU、sklearn CPU。Windows 原生更容易走 XGBoost GPU；cuML 推荐 WSL2/Linux。", wraplength=300, foreground="#555").grid(row=r, column=0, columnspan=2, sticky="w", pady=8)

        r = 0
        for label, var in [
            ("序列长度 seq_len", self.lnn_seq_len_var),
            ("隐藏层 hidden", self.lnn_hidden_var),
            ("训练轮数 epochs", self.lnn_epochs_var),
            ("批大小 batch", self.lnn_batch_var),
            ("学习率 lr", self.lnn_lr_var),
            ("dropout", self.lnn_dropout_var),
        ]:
            ttk.Label(page3, text=label + "：").grid(row=r, column=0, sticky="w", pady=5)
            ttk.Entry(page3, textvariable=var, width=18).grid(row=r, column=1, sticky="w", pady=5)
            r += 1
        ttk.Checkbutton(page3, text="使用 RTX 4090D", variable=self.use_4090_var).grid(row=r, column=0, columnspan=2, sticky="w", pady=4)
        r += 1
        ttk.Checkbutton(page3, text="使用 RTX 5090D", variable=self.use_5090_var).grid(row=r, column=0, columnspan=2, sticky="w", pady=4)
        r += 1
        ttk.Checkbutton(page3, text="双GPU并行：5090D主卡+4090D协同", variable=self.use_dual_gpu_parallel_var).grid(row=r, column=0, columnspan=2, sticky="w", pady=4)
        r += 1
        for label, var in [
            ("主卡编号", self.primary_gpu_var),
            ("协同卡编号", self.secondary_gpu_var),
        ]:
            ttk.Label(page3, text=label + "：").grid(row=r, column=0, sticky="w", pady=5)
            ttk.Entry(page3, textvariable=var, width=18).grid(row=r, column=1, sticky="w", pady=5)
            r += 1
        ttk.Checkbutton(page3, text="GPU不可用时允许CPU", variable=self.cpu_fallback_var).grid(row=r, column=0, columnspan=2, sticky="w", pady=4)
        r += 1
        ttk.Button(page3, text="检测GPU", command=self._detect_gpu).grid(row=r, column=0, columnspan=2, sticky="we", pady=8)

        r = 0
        ttk.Checkbutton(page5, text="启用低位缩量上涨/高位放量下跌修正", variable=self.use_vp_pattern_var).grid(row=r, column=0, columnspan=2, sticky="w", pady=5)
        r += 1
        for label, var in [
            ("低位阈值 position<=", self.vp_low_pos_cutoff_var),
            ("高位阈值 position>=", self.vp_high_pos_cutoff_var),
            ("缩量阈值 vol_ratio<=", self.vp_shrink_cutoff_var),
            ("放量阈值 vol_ratio>=", self.vp_expand_cutoff_var),
            ("低位缩量上涨加分", self.low_volume_up_bonus_var),
            ("高位放量下跌扣分", self.high_volume_down_penalty_var),
        ]:
            ttk.Label(page5, text=label + "：").grid(row=r, column=0, sticky="w", pady=5)
            ttk.Entry(page5, textvariable=var, width=18).grid(row=r, column=1, sticky="w", pady=5)
            r += 1
        ttk.Label(
            page5,
            text="position 表示股票在20/60日高低区间中的位置：0接近低位，1接近高位。该模块既作为模型特征，也可对最终分数做轻微修正。",
            wraplength=310, foreground="#555"
        ).grid(row=r, column=0, columnspan=2, sticky="w", pady=8)

        r = 0
        ttk.Checkbutton(page6, text="启用三日回测自动选参", variable=self.auto_param_search_var).grid(row=r, column=0, columnspan=2, sticky="w", pady=5)
        r += 1
        for label, var in [
            ("回测最近交易日数", self.param_search_days_var),
            ("每组回测TopN", self.param_search_topn_var),
            ("候选参数组数", self.param_search_candidates_var),
            ("选参最大训练行数", self.param_search_max_train_rows_var),
            ("选参最大树数量", self.param_search_max_trees_var),
            ("最少历史交易日", self.param_search_min_history_days_var),
        ]:
            ttk.Label(page6, text=label + "：").grid(row=r, column=0, sticky="w", pady=5)
            ttk.Entry(page6, textvariable=var, width=18).grid(row=r, column=1, sticky="w", pady=5)
            r += 1
        ttk.Label(
            page6,
            text="逻辑：先用最近N个已知交易日做滚动预测，分别检验下一日涨停/大涨命中与排序质量；每组参数得到可靠度分数，最终用分数最高的参数重新训练并预测下一交易日。选参阶段使用RF/ExtraTrees快速代理，最终预测仍按你在“任务”页选择的路径执行。",
            wraplength=310, foreground="#555"
        ).grid(row=r, column=0, columnspan=2, sticky="w", pady=8)

        r = 0
        ttk.Checkbutton(page7, text="启用分池输出", variable=self.use_pool_output_var).grid(row=r, column=0, columnspan=2, sticky="w", pady=5)
        r += 1
        for label, var in [
            ("每个分池TopN", self.pool_topn_each_var),
            ("自动选参涨停权重上限", self.max_auto_weight_limit_var),
            ("启动池最低涨幅%", self.startup_min_pct_var),
            ("潜伏池最低涨幅%", self.latent_min_pct_var),
            ("潜伏池最高涨幅%", self.latent_max_pct_var),
            ("潜伏池低位信号阈值", self.latent_low_signal_min_var),
            ("风险池高位风险阈值", self.risk_high_signal_min_var),
        ]:
            ttk.Label(page7, text=label + "：").grid(row=r, column=0, sticky="w", pady=5)
            ttk.Entry(page7, textvariable=var, width=18).grid(row=r, column=1, sticky="w", pady=5)
            r += 1
        ttk.Label(
            page7,
            text="分池逻辑：强势延续池看今日涨停后的连板/继续大涨；启动观察池看今日2%以上但未涨停的加速可能；低位潜伏池专门保留低位缩量上涨信号；风险排除池把高位放量下跌单独列出，避免混入稳健主榜。自动选参涨停权重上限用于防止过拟合最近连板行情。",
            wraplength=310, foreground="#555"
        ).grid(row=r, column=0, columnspan=2, sticky="w", pady=8)
        r += 1
        ttk.Checkbutton(page7, text="启用大盘/板块预测分支", variable=self.use_market_sector_branch_var).grid(row=r, column=0, columnspan=2, sticky="w", pady=5)
        r += 1
        ttk.Checkbutton(page7, text="启用对抗验证/漂移校准", variable=self.use_adversarial_validation_var).grid(row=r, column=0, columnspan=2, sticky="w", pady=5)
        r += 1
        for label, var in [
            ("漂移扣分强度", self.drift_penalty_strength_var),
            ("板块汇总TopN", self.sector_topn_var),
        ]:
            ttk.Label(page7, text=label + "：").grid(row=r, column=0, sticky="w", pady=5)
            ttk.Entry(page7, textvariable=var, width=18).grid(row=r, column=1, sticky="w", pady=5)
            r += 1

        r = 0
        ttk.Label(page4, text="外部运算脚本：").grid(row=r, column=0, sticky="w", pady=5)
        r += 1
        ttk.Entry(page4, textvariable=self.external_script_var, width=34).grid(row=r, column=0, sticky="we", pady=5)
        ttk.Button(page4, text="选择", command=self._browse_script).grid(row=r, column=1, padx=4)
        r += 1
        ttk.Button(page4, text="运行外部脚本", command=self._run_external_script).grid(row=r, column=0, columnspan=2, sticky="we", pady=8)
        r += 1
        ttk.Label(page4, text="接口参数：--db 当前数据库 --output 输出目录。也会设置 STOCK_DB_PATH 和 STOCK_OUTPUT_DIR 环境变量。", wraplength=310, foreground="#555").grid(row=r, column=0, columnspan=2, sticky="w", pady=8)

    def _build_logs_and_results(self, parent):
        upper = ttk.LabelFrame(parent, text="运行日志")
        upper.pack(fill=tk.BOTH, expand=True, pady=(0, 8))
        self.log_text = tk.Text(upper, height=16, wrap="word")
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb = ttk.Scrollbar(upper, command=self.log_text.yview)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.configure(yscrollcommand=sb.set)

        lower = ttk.LabelFrame(parent, text="候选结果预览")
        lower.pack(fill=tk.BOTH, expand=True)
        self.result_nb = ttk.Notebook(lower)
        self.result_nb.pack(fill=tk.BOTH, expand=True)
        self.tree_low = self._create_result_tree(self.result_nb, "低价候选")
        self.tree_high = self._create_result_tree(self.result_nb, "高价候选")
        self.tree_all = self._create_result_tree(self.result_nb, "综合Top")
        self.tree_continuation = self._create_result_tree(self.result_nb, "强势延续池")
        self.tree_startup = self._create_result_tree(self.result_nb, "启动观察池")
        self.tree_latent = self._create_result_tree(self.result_nb, "低位潜伏池")
        self.tree_risk = self._create_result_tree(self.result_nb, "风险排除池")
        self.tree_market = self._create_summary_tree(self.result_nb, "大盘/市场")
        self.tree_sector = self._create_summary_tree(self.result_nb, "行业板块")

    def _create_result_tree(self, nb, title: str):
        frame = ttk.Frame(nb)
        nb.add(frame, text=title)
        cols = [
            "ts_code", "name", "industry", "close",
            "low_volume_up_signal", "high_volume_down_signal", "vp_adjustment",
            "rf_limit_prob", "rf_bigrise_prob", "lnn_limit_prob", "lnn_bigrise_prob",
            "continuation_score", "startup_score", "latent_score", "risk_score", "final_score"
        ]
        tree = ttk.Treeview(frame, columns=cols, show="headings", height=12)
        widths = [90, 90, 110, 70, 120, 130, 100, 100, 100, 100, 100, 110, 100, 90, 80, 90]
        for c, w in zip(cols, widths):
            tree.heading(c, text=c)
            tree.column(c, width=w, anchor="center")
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb = ttk.Scrollbar(frame, command=tree.yview)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        tree.configure(yscrollcommand=sb.set)
        return tree

    def _create_summary_tree(self, nb, title: str):
        frame = ttk.Frame(nb)
        nb.add(frame, text=title)
        cols = [
            "view_type", "group_name", "direction", "composite_score", "risk_score",
            "avg_limit_prob", "avg_bigrise_prob", "breadth_up_pct", "avg_pct_chg_today",
            "stock_count", "reason", "top_codes"
        ]
        widths = [90, 120, 90, 105, 80, 105, 105, 90, 95, 75, 260, 260]
        tree = ttk.Treeview(frame, columns=cols, show="headings", height=12)
        for c, w in zip(cols, widths):
            tree.heading(c, text=c)
            tree.column(c, width=w, anchor="center")
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb = ttk.Scrollbar(frame, command=tree.yview)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        tree.configure(yscrollcommand=sb.set)
        return tree

    def _browse_db(self):
        p = filedialog.askopenfilename(title="选择 SQLite 数据库", filetypes=[("SQLite DB", "*.db *.sqlite *.sqlite3"), ("All", "*.*")])
        if p:
            self.db_var.set(p)

    def _browse_out(self):
        p = filedialog.askdirectory(title="选择输出目录")
        if p:
            self.out_var.set(p)

    def _browse_script(self):
        p = filedialog.askopenfilename(title="选择外部 Python 运算脚本", filetypes=[("Python", "*.py"), ("All", "*.*")])
        if p:
            self.external_script_var.set(p)

    def _detect_gpu(self):
        try:
            import torch
            if not torch.cuda.is_available():
                messagebox.showinfo("GPU检测", "未检测到 CUDA GPU。")
                return
            lines = [f"cuda:{i} - {torch.cuda.get_device_name(i)}" for i in range(torch.cuda.device_count())]
            messagebox.showinfo("GPU检测", "\n".join(lines))
        except Exception as e:
            messagebox.showerror("GPU检测失败", str(e))

    def _parse_config(self) -> AppConfig:
        def f(var, default=0.0):
            try:
                return float(var.get())
            except Exception:
                return default
        def i(var, default=0):
            try:
                return int(float(var.get()))
            except Exception:
                return default
        cfg = AppConfig(
            db_path=self.db_var.get().strip(),
            output_dir=self.out_var.get().strip(),
            start_date=self.start_date_var.get().strip(),
            exclude_bj=self.exclude_bj_var.get(),
            exclude_st=self.exclude_st_var.get(),
            min_amount=f(self.min_amount_var, 0),
            min_close=f(self.min_close_var, 1),
            max_close=f(self.max_close_var, 300),
            low_price_cutoff=f(self.low_price_cutoff_var, 20),
            limit_threshold=f(self.limit_threshold_var, 9.5),
            bigrise_threshold=f(self.bigrise_threshold_var, 5),
            final_topn_each_band=i(self.final_topn_var, 100),
            first_stage_topn=i(self.first_stage_topn_var, 800),
            max_train_rows=i(self.max_train_rows_var, 250000),
            feature_history_days=i(self.feature_history_days_var, 900),
            pathway=self.pathway_var.get(),
            tree_model=self.tree_model_var.get(),
            tree_backend=self.tree_backend_var.get(),
            rf_n_estimators=i(self.rf_n_estimators_var, 600),
            rf_max_depth=i(self.rf_max_depth_var, 10),
            rf_min_samples_leaf=i(self.rf_min_leaf_var, 20),
            rf_n_jobs=i(self.rf_n_jobs_var, -1),
            rf_class_weight=self.rf_class_weight_var.get(),
            lnn_seq_len=i(self.lnn_seq_len_var, 20),
            lnn_hidden_size=i(self.lnn_hidden_var, 96),
            lnn_epochs=i(self.lnn_epochs_var, 8),
            lnn_batch_size=i(self.lnn_batch_var, 1024),
            lnn_lr=f(self.lnn_lr_var, 1e-3),
            lnn_dropout=f(self.lnn_dropout_var, 0.1),
            use_4090d=self.use_4090_var.get(),
            use_5090d=self.use_5090_var.get(),
            allow_cpu_fallback=self.cpu_fallback_var.get(),
            preferred_primary_gpu=i(self.primary_gpu_var, 0),
            preferred_secondary_gpu=i(self.secondary_gpu_var, 1),
            use_dual_gpu_parallel=self.use_dual_gpu_parallel_var.get(),
            weight_limit=f(self.weight_limit_var, 0.65),
            weight_bigrise=f(self.weight_bigrise_var, 0.35),
            use_vp_pattern=self.use_vp_pattern_var.get(),
            vp_low_pos_cutoff=f(self.vp_low_pos_cutoff_var, 0.35),
            vp_high_pos_cutoff=f(self.vp_high_pos_cutoff_var, 0.70),
            vp_shrink_cutoff=f(self.vp_shrink_cutoff_var, 0.80),
            vp_expand_cutoff=f(self.vp_expand_cutoff_var, 1.50),
            low_volume_up_bonus=f(self.low_volume_up_bonus_var, 0.06),
            high_volume_down_penalty=f(self.high_volume_down_penalty_var, 0.10),
            auto_param_search=self.auto_param_search_var.get(),
            param_search_days=i(self.param_search_days_var, 3),
            param_search_topn=i(self.param_search_topn_var, 80),
            param_search_candidates=i(self.param_search_candidates_var, 4),
            param_search_max_train_rows=i(self.param_search_max_train_rows_var, 60000),
            param_search_max_trees=i(self.param_search_max_trees_var, 220),
            param_search_min_history_days=i(self.param_search_min_history_days_var, 120),
            use_pool_output=self.use_pool_output_var.get(),
            pool_topn_each=i(self.pool_topn_each_var, 30),
            max_auto_weight_limit=f(self.max_auto_weight_limit_var, 0.70),
            startup_min_pct=f(self.startup_min_pct_var, 2.0),
            latent_min_pct=f(self.latent_min_pct_var, 0.0),
            latent_max_pct=f(self.latent_max_pct_var, 5.0),
            latent_low_signal_min=f(self.latent_low_signal_min_var, 0.05),
            risk_high_signal_min=f(self.risk_high_signal_min_var, 0.25),
            use_market_sector_branch=self.use_market_sector_branch_var.get(),
            use_adversarial_validation=self.use_adversarial_validation_var.get(),
            drift_penalty_strength=f(self.drift_penalty_strength_var, 0.035),
            sector_topn=i(self.sector_topn_var, 60),
        )
        if not cfg.db_path:
            raise ValueError("请填写数据库路径。")
        if not os.path.exists(cfg.db_path):
            raise ValueError(f"数据库不存在：{cfg.db_path}")
        if cfg.weight_limit + cfg.weight_bigrise <= 0:
            raise ValueError("涨停权重 + 大涨权重必须大于0。")
        s = cfg.weight_limit + cfg.weight_bigrise
        cfg.weight_limit = cfg.weight_limit / s
        cfg.weight_bigrise = cfg.weight_bigrise / s
        cfg.max_auto_weight_limit = min(max(float(cfg.max_auto_weight_limit), 0.50), 0.95)
        cfg.pool_topn_each = max(1, int(cfg.pool_topn_each))
        return cfg

    def _start(self):
        if self.worker and self.worker.is_alive():
            messagebox.showwarning("任务运行中", "当前已有任务正在运行。")
            return
        try:
            cfg = self._parse_config()
        except Exception as e:
            messagebox.showerror("参数错误", str(e))
            return
        self.stop_event.clear()
        self.progress_var.set(0)
        self.status_var.set("启动中")
        self._clear_trees()
        self._log("=" * 80)
        self._log("启动预测任务")
        self._log(json.dumps(asdict(cfg), ensure_ascii=False, indent=2))
        self.worker = threading.Thread(target=self._run_worker, args=(cfg,), daemon=True)
        self.worker.start()

    def _run_worker(self, cfg: AppConfig):
        try:
            engine = PredictionEngine(cfg, self._thread_log, self._thread_progress, self.stop_event)
            result = engine.run()
            self.log_queue.put(("result", result))
        except Exception as e:
            tb = traceback.format_exc()
            self.log_queue.put(("error", f"{e}\n\n{tb}"))

    def _stop(self):
        self.stop_event.set()
        self._log("已请求停止；当前批次结束后会退出。")
        self.status_var.set("正在停止")

    def _thread_log(self, msg: str):
        self.log_queue.put(("log", msg))

    def _thread_progress(self, value: int, status: str):
        self.log_queue.put(("progress", value, status))

    def _poll_queue(self):
        try:
            while True:
                item = self.log_queue.get_nowait()
                kind = item[0]
                if kind == "log":
                    self._log(item[1])
                elif kind == "progress":
                    self.progress_var.set(int(item[1]))
                    self.status_var.set(str(item[2]))
                elif kind == "error":
                    self._log(item[1])
                    self.status_var.set("失败")
                    messagebox.showerror("运行失败", item[1][:4000])
                elif kind == "result":
                    self.current_result = item[1]
                    self._show_result(item[1])
                    self.status_var.set("完成")
                    messagebox.showinfo("完成", f"预测完成。\n输出：{item[1].get('csv_all', '')}")
        except Exception:
            pass
        self.after(150, self._poll_queue)

    def _log(self, msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{ts}] {msg}\n")
        self.log_text.see(tk.END)

    def _clear_trees(self):
        for tree in [self.tree_low, self.tree_high, self.tree_all, self.tree_continuation, self.tree_startup, self.tree_latent, self.tree_risk, self.tree_market, self.tree_sector]:
            for item in tree.get_children():
                tree.delete(item)

    def _show_result(self, result: Dict[str, object]):
        self._clear_trees()
        self._fill_tree(self.tree_low, result.get("low_df"))
        self._fill_tree(self.tree_high, result.get("high_df"))
        self._fill_tree(self.tree_all, result.get("all_top"))
        self._fill_tree(self.tree_continuation, result.get("strong_continuation_df"))
        self._fill_tree(self.tree_startup, result.get("startup_watch_df"))
        self._fill_tree(self.tree_latent, result.get("low_latent_df"))
        self._fill_tree(self.tree_risk, result.get("risk_exclusion_df"))
        self._fill_summary_tree(self.tree_market, result.get("market_summary_df"))
        self._fill_summary_tree(self.tree_sector, result.get("sector_summary_df"))

    def _fill_tree(self, tree, df_obj):
        if df_obj is None:
            return
        df = df_obj.copy()
        cols = [
            "ts_code", "name", "industry", "close",
            "low_volume_up_signal", "high_volume_down_signal", "vp_adjustment",
            "rf_limit_prob", "rf_bigrise_prob", "lnn_limit_prob", "lnn_bigrise_prob",
            "continuation_score", "startup_score", "latent_score", "risk_score", "final_score"
        ]
        for _, row in df.head(200).iterrows():
            vals = []
            for c in cols:
                v = row.get(c, "")
                if isinstance(v, float):
                    vals.append(f"{v:.4f}")
                else:
                    vals.append(str(v))
            tree.insert("", tk.END, values=vals)

    def _fill_summary_tree(self, tree, df_obj):
        if df_obj is None:
            return
        df = df_obj.copy()
        cols = [
            "view_type", "group_name", "direction", "composite_score", "risk_score",
            "avg_limit_prob", "avg_bigrise_prob", "breadth_up_pct", "avg_pct_chg_today",
            "stock_count", "reason", "top_codes"
        ]
        for _, row in df.head(300).iterrows():
            vals = []
            for c in cols:
                v = row.get(c, "")
                if isinstance(v, float):
                    vals.append(f"{v:.4f}")
                else:
                    vals.append(str(v))
            tree.insert("", tk.END, values=vals)

    def _save_config(self):
        try:
            cfg = self._parse_config()
            p = filedialog.asksaveasfilename(title="保存参数", defaultextension=".json", filetypes=[("JSON", "*.json")])
            if p:
                with open(p, "w", encoding="utf-8") as f:
                    json.dump(asdict(cfg), f, ensure_ascii=False, indent=2)
                messagebox.showinfo("已保存", p)
        except Exception as e:
            messagebox.showerror("保存失败", str(e))

    def _load_config(self):
        p = filedialog.askopenfilename(title="读取参数", filetypes=[("JSON", "*.json"), ("All", "*.*")])
        if not p:
            return
        try:
            with open(p, "r", encoding="utf-8") as f:
                d = json.load(f)
            self.db_var.set(d.get("db_path", self.db_var.get()))
            self.out_var.set(d.get("output_dir", self.out_var.get()))
            self.start_date_var.set(d.get("start_date", self.start_date_var.get()))
            self.pathway_var.set(d.get("pathway", self.pathway_var.get()))
            self.tree_model_var.set(d.get("tree_model", self.tree_model_var.get()))
            self.tree_backend_var.set(d.get("tree_backend", self.tree_backend_var.get()))
            self.exclude_bj_var.set(bool(d.get("exclude_bj", True)))
            self.exclude_st_var.set(bool(d.get("exclude_st", True)))
            self.use_4090_var.set(bool(d.get("use_4090d", True)))
            self.use_5090_var.set(bool(d.get("use_5090d", True)))
            self.cpu_fallback_var.set(bool(d.get("allow_cpu_fallback", True)))
            self.use_dual_gpu_parallel_var.set(bool(d.get("use_dual_gpu_parallel", True)))
            self.rf_class_weight_var.set(bool(d.get("rf_class_weight", True)))
            self.use_vp_pattern_var.set(bool(d.get("use_vp_pattern", True)))
            self.auto_param_search_var.set(bool(d.get("auto_param_search", True)))
            self.use_pool_output_var.set(bool(d.get("use_pool_output", True)))
            self.use_market_sector_branch_var.set(bool(d.get("use_market_sector_branch", True)))
            self.use_adversarial_validation_var.set(bool(d.get("use_adversarial_validation", True)))
            mapping = [
                ("min_amount", self.min_amount_var), ("min_close", self.min_close_var), ("max_close", self.max_close_var),
                ("low_price_cutoff", self.low_price_cutoff_var), ("limit_threshold", self.limit_threshold_var),
                ("bigrise_threshold", self.bigrise_threshold_var), ("final_topn_each_band", self.final_topn_var),
                ("first_stage_topn", self.first_stage_topn_var), ("max_train_rows", self.max_train_rows_var),
                ("feature_history_days", self.feature_history_days_var),
                ("preferred_primary_gpu", self.primary_gpu_var),
                ("preferred_secondary_gpu", self.secondary_gpu_var),
                ("rf_n_estimators", self.rf_n_estimators_var), ("rf_max_depth", self.rf_max_depth_var),
                ("rf_min_samples_leaf", self.rf_min_leaf_var), ("rf_n_jobs", self.rf_n_jobs_var),
                ("lnn_seq_len", self.lnn_seq_len_var), ("lnn_hidden_size", self.lnn_hidden_var),
                ("lnn_epochs", self.lnn_epochs_var), ("lnn_batch_size", self.lnn_batch_var),
                ("lnn_lr", self.lnn_lr_var), ("lnn_dropout", self.lnn_dropout_var),
                ("weight_limit", self.weight_limit_var), ("weight_bigrise", self.weight_bigrise_var),
                ("vp_low_pos_cutoff", self.vp_low_pos_cutoff_var),
                ("vp_high_pos_cutoff", self.vp_high_pos_cutoff_var),
                ("vp_shrink_cutoff", self.vp_shrink_cutoff_var),
                ("vp_expand_cutoff", self.vp_expand_cutoff_var),
                ("low_volume_up_bonus", self.low_volume_up_bonus_var),
                ("high_volume_down_penalty", self.high_volume_down_penalty_var),
                ("pool_topn_each", self.pool_topn_each_var),
                ("max_auto_weight_limit", self.max_auto_weight_limit_var),
                ("startup_min_pct", self.startup_min_pct_var),
                ("latent_min_pct", self.latent_min_pct_var),
                ("latent_max_pct", self.latent_max_pct_var),
                ("latent_low_signal_min", self.latent_low_signal_min_var),
                ("risk_high_signal_min", self.risk_high_signal_min_var),
                ("param_search_days", self.param_search_days_var),
                ("param_search_topn", self.param_search_topn_var),
                ("param_search_candidates", self.param_search_candidates_var),
                ("param_search_max_train_rows", self.param_search_max_train_rows_var),
                ("param_search_max_trees", self.param_search_max_trees_var),
                ("param_search_min_history_days", self.param_search_min_history_days_var),
                ("drift_penalty_strength", self.drift_penalty_strength_var),
                ("sector_topn", self.sector_topn_var),
            ]
            for k, var in mapping:
                if k in d:
                    var.set(str(d[k]))
            messagebox.showinfo("已读取", p)
        except Exception as e:
            messagebox.showerror("读取失败", str(e))

    def _run_external_script(self):
        script = self.external_script_var.get().strip()
        if not script or not os.path.exists(script):
            messagebox.showerror("脚本不存在", "请先选择外部 Python 脚本。")
            return
        db = self.db_var.get().strip()
        out = self.out_var.get().strip()
        Path(out).mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env["STOCK_DB_PATH"] = db
        env["STOCK_OUTPUT_DIR"] = out
        cmd = [sys.executable, script, "--db", db, "--output", out]
        self._log("运行外部脚本：" + " ".join(cmd))

        def worker():
            try:
                p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", env=env)
                assert p.stdout is not None
                for line in p.stdout:
                    self.log_queue.put(("log", line.rstrip()))
                code = p.wait()
                if code != 0:
                    self.log_queue.put(("error", f"外部脚本退出码：{code}"))
                else:
                    self.log_queue.put(("log", "外部脚本完成。"))
            except Exception as e:
                self.log_queue.put(("error", traceback.format_exc()))

        threading.Thread(target=worker, daemon=True).start()


def main():
    # 避免 Windows 控制台 GBK 对少数字符报错；GUI 本身不依赖控制台输出。
    os.environ.setdefault("PYTHONUTF8", "1")
    app = StockPredictorGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
