# -*- coding: utf-8 -*-
"""
诊断反馈缓冲库为什么没有产生回填/梯度。

默认读取：
    F:\\股票\\ml_feedback_buffer.db
    F:\\股票\\tu_share_data.db

本脚本只读库，不写入、不修改数据。
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path
from typing import Iterable, List, Optional, Tuple


DEFAULT_BUFFER_DB = r"ml_feedback_buffer.db"
DEFAULT_DATA_DB = r"data\tu_share_data.db"


def q(conn: sqlite3.Connection, sql: str, params: Iterable = ()) -> list:
    try:
        return conn.execute(sql, tuple(params)).fetchall()
    except Exception as e:
        print(f"[SQL失败] {sql}\n  -> {e}")
        return []


def table_exists(conn: sqlite3.Connection, table: str) -> bool:
    rows = q(conn, "SELECT name FROM sqlite_master WHERE type='table' AND name=?", [table])
    return bool(rows)


def columns(conn: sqlite3.Connection, table: str) -> List[str]:
    return [r[1] for r in q(conn, f'PRAGMA table_info("{table}")')]


def scalar(conn: sqlite3.Connection, sql: str, params: Iterable = (), default=None):
    rows = q(conn, sql, params)
    if not rows:
        return default
    return rows[0][0]


def pick_col(cols: List[str], candidates: List[str]) -> Optional[str]:
    lower = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def fmt_date(v) -> str:
    return "" if v is None else str(v)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--buffer-db", default=DEFAULT_BUFFER_DB)
    parser.add_argument("--data-db", default=DEFAULT_DATA_DB)
    parser.add_argument("--sample-dates", type=int, default=8)
    args = parser.parse_args()

    buffer_path = Path(args.buffer_db)
    data_path = Path(args.data_db)
    print("=" * 76)
    print("反馈回填诊断")
    print("=" * 76)
    print(f"反馈库: {buffer_path}")
    print(f"行情库: {data_path}")

    if not buffer_path.exists():
        print("结论: 反馈库不存在。")
        return
    if not data_path.exists():
        print("结论: 行情库不存在。")
        return

    fb = sqlite3.connect(str(buffer_path))
    db = sqlite3.connect(str(data_path))

    required_fb = ["prediction_snapshot", "realized_predictions", "feedback_stock_gradient", "feedback_industry_gradient"]
    for t in required_fb:
        print(f"- 反馈表 {t}: {'存在' if table_exists(fb, t) else '不存在'}")
    print(f"- 行情表 daily: {'存在' if table_exists(db, 'daily') else '不存在'}")

    if not table_exists(fb, "prediction_snapshot") or not table_exists(db, "daily"):
        print("结论: 缺少 prediction_snapshot 或 daily 表，无法回填。")
        return

    snap_cols = columns(fb, "prediction_snapshot")
    daily_cols = columns(db, "daily")
    print(f"- prediction_snapshot字段: {', '.join(snap_cols)}")
    print(f"- daily字段: {', '.join(daily_cols)}")

    snap_code = pick_col(snap_cols, ["ts_code", "code", "symbol"])
    snap_date = pick_col(snap_cols, ["trade_date", "base_trade_date", "predict_trade_date", "snapshot_trade_date"])
    daily_code = pick_col(daily_cols, ["ts_code", "code", "symbol"])
    daily_date = pick_col(daily_cols, ["trade_date", "date"])
    daily_pct = pick_col(daily_cols, ["pct_chg", "pct_change", "change_pct"])

    print(f"- 识别字段: 快照代码={snap_code}, 快照日期={snap_date}, 日线代码={daily_code}, 日线日期={daily_date}, 涨跌幅={daily_pct}")
    if not all([snap_code, snap_date, daily_code, daily_date]):
        print("结论: 字段名没有识别出来，回填逻辑大概率无法对齐。")
        return

    snap_n = scalar(fb, 'SELECT COUNT(1) FROM prediction_snapshot', default=0)
    realized_n = scalar(fb, 'SELECT COUNT(1) FROM realized_predictions', default=0) if table_exists(fb, "realized_predictions") else 0
    stock_grad_n = scalar(fb, 'SELECT COUNT(1) FROM feedback_stock_gradient', default=0) if table_exists(fb, "feedback_stock_gradient") else 0
    industry_grad_n = scalar(fb, 'SELECT COUNT(1) FROM feedback_industry_gradient', default=0) if table_exists(fb, "feedback_industry_gradient") else 0
    snap_min = scalar(fb, f'SELECT MIN("{snap_date}") FROM prediction_snapshot')
    snap_max = scalar(fb, f'SELECT MAX("{snap_date}") FROM prediction_snapshot')
    daily_min = scalar(db, f'SELECT MIN("{daily_date}") FROM daily')
    daily_max = scalar(db, f'SELECT MAX("{daily_date}") FROM daily')

    print("\n一、总览")
    print(f"- 预测快照: {snap_n}")
    print(f"- 已回填真实表现: {realized_n}")
    print(f"- 股票梯度: {stock_grad_n}，行业梯度: {industry_grad_n}")
    print(f"- 快照日期范围: {fmt_date(snap_min)} -> {fmt_date(snap_max)}")
    print(f"- daily日期范围: {fmt_date(daily_min)} -> {fmt_date(daily_max)}")

    print("\n二、按快照日期检查下一交易日")
    snap_dates = q(
        fb,
        f'SELECT "{snap_date}", COUNT(1), COUNT(DISTINCT "{snap_code}") '
        f'FROM prediction_snapshot GROUP BY "{snap_date}" ORDER BY "{snap_date}" DESC LIMIT ?',
        [args.sample_dates],
    )
    any_match = False
    for d, row_n, code_n in snap_dates:
        next_date = scalar(db, f'SELECT MIN("{daily_date}") FROM daily WHERE "{daily_date}" > ?', [d])
        if next_date is None:
            print(f"- {d}: 快照{row_n}条/{code_n}只；daily里找不到后续交易日")
            continue
        next_daily_n = scalar(db, f'SELECT COUNT(1) FROM daily WHERE "{daily_date}"=?', [next_date], default=0)
        snap_codes = set(r[0] for r in q(fb, f'SELECT DISTINCT "{snap_code}" FROM prediction_snapshot WHERE "{snap_date}"=?', [d]))
        daily_codes = set(r[0] for r in q(db, f'SELECT DISTINCT "{daily_code}" FROM daily WHERE "{daily_date}"=?', [next_date]))
        hit_n = len(snap_codes & daily_codes)
        any_match = any_match or hit_n > 0
        print(f"- {d}: 下一交易日={next_date}；快照{row_n}条/{code_n}只；daily当日{next_daily_n}条；代码命中{hit_n}只")

    print("\n三、判断")
    if realized_n > 0 and (stock_grad_n > 0 or industry_grad_n > 0):
        print("- 反馈已经生效。")
    elif snap_n == 0:
        print("- 没有预测快照，所以无法回填。")
    elif daily_max is None:
        print("- daily没有数据，所以无法回填。")
    elif str(daily_max) <= str(snap_min):
        print("- daily最大日期不晚于快照最早日期，说明真实下一交易日数据还没入库。")
    elif not any_match:
        print("- daily已有后续日期，但股票代码完全匹配不上，重点检查 ts_code 格式是否一致。")
    else:
        print("- daily已有后续日期且代码能匹配，但 realized 仍为0：大概率是原训练脚本的回填SQL/字段名对齐逻辑有问题。")
        print("- 这种情况会让反馈缓冲不生效，但不影响原始模型正常预测。需要升级训练脚本里的回填函数。")

    fb.close()
    db.close()


if __name__ == "__main__":
    main()
