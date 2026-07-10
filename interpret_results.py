# -*- coding: utf-8 -*-
"""
三次预测 + 反馈缓冲结果解读脚本。

默认读取：
    F:\\股票\\ml_outputs
    F:\\股票\\ml_feedback_buffer.db

输出：
    1) 三次预测共振后的重点关注梯队
    2) 下一交易日大盘/市场三次融合判断
    3) 行业板块三次融合强弱
    4) 反馈缓冲库状态
    5) 本次运行需要处理的问题

用法：
    python interpret_three_run_buffered_results.py
    python interpret_three_run_buffered_results.py --top 18
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


DEFAULT_OUTPUT_DIR = r"ml_outputs"
DEFAULT_BUFFER_DB = r"ml_feedback_buffer.db"


def latest_files(output_dir: str, pattern: str, n: int = 3) -> List[str]:
    files = glob.glob(str(Path(output_dir) / pattern))
    files = sorted(files, key=lambda p: os.path.getmtime(p), reverse=True)
    return list(reversed(files[:n]))


def read_csv(path: str) -> pd.DataFrame:
    for enc in ["utf-8-sig", "utf-8", "gbk"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            pass
    return pd.DataFrame()


def run_id_from_path(path: str) -> str:
    m = re.search(r"(\d{8}_\d{6})", Path(path).name)
    return m.group(1) if m else Path(path).stem


def fmt(v, digits: int = 4) -> str:
    try:
        if pd.isna(v):
            return ""
        return f"{float(v):.{digits}f}"
    except Exception:
        return str(v)


def normalize_bool_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower().isin(["true", "1", "yes", "y"])


def load_three_results(output_dir: str) -> pd.DataFrame:
    frames = []
    for path in latest_files(output_dir, "nextday_limit_lnn_rf_all_*.csv", 3):
        df = read_csv(path)
        if df.empty:
            continue
        df["source_file"] = path
        df["run_id"] = run_id_from_path(path)
        df["rank_in_run"] = pd.to_numeric(df.get("final_score", 0), errors="coerce").rank(ascending=False, method="min")
        frames.append(df)
    return pd.concat(frames, axis=0, ignore_index=True) if frames else pd.DataFrame()


def load_three_pools(output_dir: str) -> pd.DataFrame:
    frames = []
    for path in latest_files(output_dir, "nextday_limit_lnn_rf_pools_*.csv", 3):
        df = read_csv(path)
        if df.empty:
            continue
        df["run_id"] = run_id_from_path(path)
        frames.append(df)
    return pd.concat(frames, axis=0, ignore_index=True) if frames else pd.DataFrame()


def pool_hint_map(pools: pd.DataFrame) -> Dict[str, str]:
    if pools.empty or "ts_code" not in pools.columns:
        return {}
    priority = {
        "强势延续池": 4,
        "启动观察池": 3,
        "低位潜伏池": 2,
        "风险排除池": -4,
        "strong_continuation": 4,
        "startup_watch": 3,
        "low_latent": 2,
        "risk_exclusion": -4,
    }
    out: Dict[str, str] = {}
    score: Dict[str, int] = {}
    for _, row in pools.iterrows():
        code = str(row.get("ts_code", ""))
        name = str(row.get("pool_name", row.get("pool_key", "")))
        p = max([v for k, v in priority.items() if k in name], default=0)
        if code and p >= score.get(code, -99):
            score[code] = p
            out[code] = name
    return out


def build_consensus(results: pd.DataFrame, pools: pd.DataFrame, top_n: int) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame()
    df = results.copy()
    for col in ["final_score", "high_volume_down_signal", "adversarial_drift_score", "drift_penalty", "feedback_adjustment"]:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    if "stage_selected" in df.columns and normalize_bool_series(df["stage_selected"]).any():
        df = df[normalize_bool_series(df["stage_selected"])].copy()

    meta_cols = [c for c in ["name", "industry", "market", "price_band"] if c in df.columns]
    agg = df.groupby("ts_code").agg(
        appear_count=("run_id", "nunique"),
        mean_final=("final_score", "mean"),
        max_final=("final_score", "max"),
        min_final=("final_score", "min"),
        std_final=("final_score", "std"),
        avg_rank=("rank_in_run", "mean"),
        best_rank=("rank_in_run", "min"),
        avg_high_risk=("high_volume_down_signal", "mean"),
        avg_drift=("adversarial_drift_score", "mean"),
        avg_drift_penalty=("drift_penalty", "mean"),
        avg_feedback=("feedback_adjustment", "mean"),
    ).reset_index()
    agg["std_final"] = agg["std_final"].fillna(0.0)
    for col in meta_cols:
        meta = df.groupby("ts_code")[col].agg(lambda s: s.dropna().astype(str).iloc[-1] if len(s.dropna()) else "").reset_index()
        agg = agg.merge(meta, on="ts_code", how="left")

    hints = pool_hint_map(pools)
    agg["pool_hint"] = agg["ts_code"].astype(str).map(hints).fillna("")
    pool_bonus = agg["pool_hint"].apply(lambda x: 0.04 if "强势" in x else (0.03 if "启动" in x else (0.02 if "潜伏" in x else (-0.08 if "风险" in x else 0.0))))
    agg["consensus_score"] = (
        agg["mean_final"]
        + 0.025 * agg["appear_count"].clip(0, 3)
        + pool_bonus
        - 0.050 * agg["std_final"].fillna(0)
        - 0.080 * agg["avg_high_risk"].clip(0, 3)
        - 0.300 * agg["avg_drift_penalty"].clip(0, 0.2)
    )
    agg["reason"] = agg.apply(reason_for_stock, axis=1)
    return agg.sort_values(["consensus_score", "appear_count", "mean_final"], ascending=False).head(top_n).reset_index(drop=True)


def reason_for_stock(row: pd.Series) -> str:
    reasons = []
    if int(row.get("appear_count", 0)) >= 3:
        reasons.append("三次预测均入选")
    elif int(row.get("appear_count", 0)) == 2:
        reasons.append("两次预测共振")
    if float(row.get("mean_final", 0)) >= 0.80:
        reasons.append("平均分高")
    if float(row.get("std_final", 0)) <= 0.04:
        reasons.append("分数稳定")
    if "强势" in str(row.get("pool_hint", "")):
        reasons.append("强势延续池")
    if float(row.get("avg_high_risk", 0)) >= 0.25:
        reasons.append("高位风险需留意")
    if float(row.get("avg_feedback", 0)) > 0:
        reasons.append("反馈缓冲加分")
    return "；".join(reasons[:4]) if reasons else "综合排序靠前"


def tier_name(rank: int) -> str:
    if rank <= 5:
        return "第一梯队"
    if rank <= 12:
        return "第二梯队"
    return "观察梯队"


def tier_desc(name: str) -> str:
    if name == "第一梯队":
        return "三次融合后最靠前，优先盯开盘强度、板块共振和承接。"
    if name == "第二梯队":
        return "强度仍高，但更需要盘中确认或等板块继续发酵。"
    return "有信号但确定性略弱，适合观察，不宜无脑追高。"


def load_summary(output_dir: str, pattern: str) -> pd.DataFrame:
    frames = []
    for path in latest_files(output_dir, pattern, 3):
        df = read_csv(path)
        if df.empty:
            continue
        df["run_id"] = run_id_from_path(path)
        frames.append(df)
    return pd.concat(frames, axis=0, ignore_index=True) if frames else pd.DataFrame()


def summarize_groups(df: pd.DataFrame, top_n: int = 8) -> pd.DataFrame:
    if df.empty or "group_name" not in df.columns:
        return pd.DataFrame()
    df = df.copy()
    bad_names = {"", "nan", "none", "null", "未知", "其他", "未分类"}
    df["group_name"] = df["group_name"].astype(str).str.strip()
    df = df[~df["group_name"].str.lower().isin(bad_names)].copy()
    if df.empty:
        return pd.DataFrame()
    for col in ["composite_score", "risk_score", "avg_limit_prob", "avg_bigrise_prob"]:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    agg = df.groupby("group_name").agg(
        appear_count=("run_id", "nunique"),
        mean_score=("composite_score", "mean"),
        mean_risk=("risk_score", "mean"),
        best_score=("composite_score", "max"),
        direction=("direction", lambda s: s.astype(str).mode().iloc[0] if len(s.mode()) else ""),
        reason=("reason", lambda s: s.dropna().astype(str).iloc[-1] if len(s.dropna()) else ""),
        top_codes=("top_codes", lambda s: s.dropna().astype(str).iloc[-1] if len(s.dropna()) else ""),
    ).reset_index()
    agg["group_score"] = agg["mean_score"] + 0.02 * agg["appear_count"].clip(0, 3) - 0.10 * agg["mean_risk"]
    return agg.sort_values("group_score", ascending=False).head(top_n).reset_index(drop=True)


def feedback_status(buffer_db: str) -> List[str]:
    if not os.path.exists(buffer_db):
        return [f"反馈缓冲库尚不存在：{buffer_db}"]
    try:
        conn = sqlite3.connect(buffer_db)
        cur = conn.cursor()
        tables = ["feedback_runs", "prediction_snapshot", "realized_predictions", "feedback_stock_gradient", "feedback_industry_gradient"]
        counts = {}
        for t in tables:
            try:
                cur.execute(f'SELECT COUNT(1) FROM "{t}"')
                counts[t] = int(cur.fetchone()[0])
            except Exception:
                counts[t] = 0
        detail = {}
        try:
            cur.execute("SELECT MIN(trade_date), MAX(trade_date), COUNT(DISTINCT run_id) FROM prediction_snapshot")
            detail["snapshot_range"] = cur.fetchone()
        except Exception:
            detail["snapshot_range"] = None
        try:
            cur.execute("SELECT MAX(realized_date) FROM realized_predictions")
            detail["max_realized_date"] = cur.fetchone()[0]
        except Exception:
            detail["max_realized_date"] = None
        conn.close()
        lines = [
            f"反馈运行数: {counts['feedback_runs']}",
            f"预测快照: {counts['prediction_snapshot']}",
            f"已回填真实表现: {counts['realized_predictions']}",
            f"股票梯度: {counts['feedback_stock_gradient']}，行业梯度: {counts['feedback_industry_gradient']}",
        ]
        if detail.get("snapshot_range"):
            mn, mx, run_n = detail["snapshot_range"]
            lines.append(f"预测快照覆盖交易日: {mn} -> {mx}，覆盖运行: {run_n}")
        if detail.get("max_realized_date"):
            lines.append(f"最新已回填真实交易日: {detail['max_realized_date']}")
        if counts["realized_predictions"] == 0:
            lines.append("提示：目前还没有真实表现回填。常见原因是预测快照对应的下一交易日行情还未入库，或脚本没有把 trade_date 与 next_trade_date 成功对齐。")
            lines.append("检查方法：确认 daily 表里已经有预测快照日期之后的下一个交易日数据，例如 20260630 的预测通常要等 20260701 行情入库后才会回填。")
        return lines
    except Exception as e:
        return [f"反馈缓冲库读取失败：{e}"]


def health_notes(results: pd.DataFrame, consensus: pd.DataFrame, market: pd.DataFrame, sector: pd.DataFrame) -> List[str]:
    notes = []
    run_count = results["run_id"].nunique() if not results.empty and "run_id" in results.columns else 0
    if run_count < 3:
        notes.append(f"只读取到 {run_count} 次预测，三次融合可信度会下降。")
    if consensus.empty:
        notes.append("没有形成个股共振名单，请检查 CSV 是否完整。")
    if market.empty:
        notes.append("没有读取到市场摘要。")
    if sector.empty:
        notes.append("没有读取到行业摘要。")
    if not results.empty and "feedback_adjustment" in results.columns:
        fb = pd.to_numeric(results["feedback_adjustment"], errors="coerce").fillna(0)
        if (fb.abs() > 1e-9).sum() == 0:
            notes.append("反馈校准本次命中为 0，通常是因为还没有下一交易日真实结果回填。")
    notes.append("cuML/cudf 未安装时自动走 XGBoost GPU，不属于失败。")
    notes.append("日志里的 GPU imbalance/NCCL warning 表示双卡不完全均衡；能跑，但若想更快可把 LNN 主卡改成 5090D 或单用 5090D。")
    return notes


def print_consensus(consensus: pd.DataFrame) -> None:
    if consensus.empty:
        print("未生成个股共振名单。")
        return
    current = ""
    for idx, row in consensus.iterrows():
        rank = idx + 1
        tier = tier_name(rank)
        if tier != current:
            current = tier
            print(f"\n{tier}：{tier_desc(tier)}")
        print(
            f"{rank:02d}. {row.get('name','')} ({row.get('ts_code','')}) | {row.get('industry','')} | "
            f"共振={fmt(row.get('consensus_score'),4)} 均分={fmt(row.get('mean_final'),4)} "
            f"出现={int(row.get('appear_count',0))}/3 波动={fmt(row.get('std_final'),4)} | "
            f"{row.get('pool_hint','')} | {row.get('reason','')}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--buffer-db", default=DEFAULT_BUFFER_DB)
    parser.add_argument("--top", type=int, default=18)
    args = parser.parse_args()

    results = load_three_results(args.output_dir)
    pools = load_three_pools(args.output_dir)
    consensus = build_consensus(results, pools, args.top)
    market = summarize_groups(load_summary(args.output_dir, "nextday_market_sector_summary_*.csv"), 8)
    sector = summarize_groups(load_summary(args.output_dir, "nextday_industry_sector_summary_*.csv"), 10)

    print("=" * 76)
    print("三次预测融合解读")
    print("=" * 76)
    if not results.empty:
        print("读取 run_id: " + ", ".join(results["run_id"].dropna().astype(str).unique().tolist()))

    print("\n一、重点关注梯队")
    print_consensus(consensus)

    print("\n二、下一交易日大盘/市场")
    if market.empty:
        print("- 未读取到市场摘要。")
    else:
        for _, row in market.iterrows():
            print(f"- {row['group_name']}: {row['direction']} | 融合={fmt(row['group_score'],3)} 均分={fmt(row['mean_score'],3)} 风险={fmt(row['mean_risk'],3)} | {row['reason']}")

    print("\n三、下一交易日行业重点")
    if sector.empty:
        print("- 未读取到行业摘要。")
    else:
        for _, row in sector.iterrows():
            print(f"- {row['group_name']}: {row['direction']} | 融合={fmt(row['group_score'],3)} 均分={fmt(row['mean_score'],3)} | {row['reason']} | 代表: {str(row['top_codes'])[:120]}")

    print("\n四、反馈缓冲库状态")
    for line in feedback_status(args.buffer_db):
        print("- " + line)

    print("\n五、本次运行需要注意")
    for line in health_notes(results, consensus, market, sector):
        print("- " + line)


if __name__ == "__main__":
    main()
