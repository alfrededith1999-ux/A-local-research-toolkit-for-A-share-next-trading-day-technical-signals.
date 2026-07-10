# -*- coding: utf-8 -*-
"""
修复反馈缓冲库回填，并重新生成股票/行业梯度。

默认读取：
    F:\\股票\\ml_feedback_buffer.db
    F:\\股票\\tu_share_data.db

作用：
    1. 从 prediction_snapshot 找到每次预测 trade_date 的下一交易日。
    2. 用 daily.pct_chg 回填 realized_predictions。
    3. 按股票、行业生成 feedback_stock_gradient / feedback_industry_gradient。

说明：
    本脚本会写入反馈缓冲库。若 realized/gradient 表结构不兼容，会先备份旧表再重建。
"""

from __future__ import annotations

import argparse
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Iterable, List


DEFAULT_BUFFER_DB = r"ml_feedback_buffer.db"
DEFAULT_DATA_DB = r"data\tu_share_data.db"


def q(conn: sqlite3.Connection, sql: str, params: Iterable = ()) -> list:
    return conn.execute(sql, tuple(params)).fetchall()


def table_exists(conn: sqlite3.Connection, table: str) -> bool:
    rows = q(conn, "SELECT name FROM sqlite_master WHERE type='table' AND name=?", [table])
    return bool(rows)


def columns(conn: sqlite3.Connection, table: str) -> List[str]:
    if not table_exists(conn, table):
        return []
    return [r[1] for r in q(conn, f'PRAGMA table_info("{table}")')]


def backup_and_drop(conn: sqlite3.Connection, table: str) -> None:
    if not table_exists(conn, table):
        return
    suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = f"{table}_bak_{suffix}"
    conn.execute(f'ALTER TABLE "{table}" RENAME TO "{backup}"')
    print(f"- 已备份旧表: {table} -> {backup}")


def ensure_normalized_tables(conn: sqlite3.Connection, force_rebuild: bool = False) -> None:
    expected = {
        "realized_predictions": {
            "run_id", "ts_code", "name", "industry", "trade_date", "realized_date",
            "final_score", "actual_pct_chg", "actual_limit", "actual_bigrise",
            "score_error", "created_at",
        },
        "feedback_stock_gradient": {
            "ts_code", "sample_count", "avg_actual_pct_chg", "hit_rate_limit",
            "hit_rate_bigrise", "avg_score", "score_adjustment", "updated_at",
        },
        "feedback_industry_gradient": {
            "industry", "sample_count", "avg_actual_pct_chg", "hit_rate_limit",
            "hit_rate_bigrise", "avg_score", "score_adjustment", "updated_at",
        },
    }

    for table, cols in expected.items():
        current = set(columns(conn, table))
        if force_rebuild or (current and not cols.issubset(current)):
            backup_and_drop(conn, table)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS realized_predictions (
            run_id TEXT NOT NULL,
            ts_code TEXT NOT NULL,
            name TEXT,
            industry TEXT,
            trade_date TEXT NOT NULL,
            realized_date TEXT NOT NULL,
            final_score REAL,
            actual_pct_chg REAL,
            actual_limit INTEGER,
            actual_bigrise INTEGER,
            score_error REAL,
            created_at TEXT,
            PRIMARY KEY (run_id, ts_code, trade_date, realized_date)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS feedback_stock_gradient (
            ts_code TEXT PRIMARY KEY,
            sample_count INTEGER,
            avg_actual_pct_chg REAL,
            hit_rate_limit REAL,
            hit_rate_bigrise REAL,
            avg_score REAL,
            score_adjustment REAL,
            updated_at TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS feedback_industry_gradient (
            industry TEXT PRIMARY KEY,
            sample_count INTEGER,
            avg_actual_pct_chg REAL,
            hit_rate_limit REAL,
            hit_rate_bigrise REAL,
            avg_score REAL,
            score_adjustment REAL,
            updated_at TEXT
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_snapshot_date_code ON prediction_snapshot(trade_date, ts_code)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_realized_date_code ON realized_predictions(trade_date, ts_code)")


def repair(buffer_db: str, data_db: str, force_rebuild: bool) -> None:
    buffer_path = Path(buffer_db)
    data_path = Path(data_db)
    if not buffer_path.exists():
        raise SystemExit(f"反馈库不存在：{buffer_path}")
    if not data_path.exists():
        raise SystemExit(f"行情库不存在：{data_path}")

    conn = sqlite3.connect(str(buffer_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")

    if not table_exists(conn, "prediction_snapshot"):
        raise SystemExit("反馈库里没有 prediction_snapshot 表。")

    ensure_normalized_tables(conn, force_rebuild=force_rebuild)

    conn.execute("ATTACH DATABASE ? AS data_db", [str(data_path)])
    before = q(conn, "SELECT COUNT(1) FROM realized_predictions")[0][0]

    print("=" * 76)
    print("开始修复反馈回填")
    print("=" * 76)
    print(f"- 修复前 realized_predictions: {before}")

    conn.execute("""
        CREATE TEMP TABLE IF NOT EXISTS tmp_next_trade_date AS
        SELECT
            p.trade_date AS trade_date,
            MIN(d.trade_date) AS realized_date
        FROM (SELECT DISTINCT trade_date FROM prediction_snapshot) p
        JOIN data_db.daily d
          ON d.trade_date > p.trade_date
        GROUP BY p.trade_date
    """)

    matched_dates = q(conn, """
        SELECT p.trade_date, n.realized_date, COUNT(1)
        FROM prediction_snapshot p
        JOIN tmp_next_trade_date n ON n.trade_date = p.trade_date
        JOIN data_db.daily d ON d.trade_date = n.realized_date AND d.ts_code = p.ts_code
        GROUP BY p.trade_date, n.realized_date
        ORDER BY p.trade_date
    """)
    print("- 可回填日期：")
    for trade_date, realized_date, cnt in matched_dates:
        print(f"  {trade_date} -> {realized_date}: {cnt} 条")

    conn.execute("""
        INSERT OR REPLACE INTO realized_predictions (
            run_id, ts_code, name, industry, trade_date, realized_date,
            final_score, actual_pct_chg, actual_limit, actual_bigrise,
            score_error, created_at
        )
        SELECT
            p.run_id,
            p.ts_code,
            p.name,
            p.industry,
            p.trade_date,
            n.realized_date,
            CAST(p.final_score AS REAL),
            CAST(d.pct_chg AS REAL),
            CASE
                WHEN p.ts_code LIKE '688%' OR p.ts_code LIKE '300%' THEN CAST(d.pct_chg AS REAL) >= 19.0
                ELSE CAST(d.pct_chg AS REAL) >= 9.5
            END AS actual_limit,
            CAST(d.pct_chg AS REAL) >= 5.0 AS actual_bigrise,
            (CAST(d.pct_chg AS REAL) / 10.0) - CAST(p.final_score AS REAL) AS score_error,
            datetime('now', 'localtime')
        FROM prediction_snapshot p
        JOIN tmp_next_trade_date n
          ON n.trade_date = p.trade_date
        JOIN data_db.daily d
          ON d.trade_date = n.realized_date
         AND d.ts_code = p.ts_code
        WHERE d.pct_chg IS NOT NULL
    """)

    conn.execute("DELETE FROM feedback_stock_gradient")
    conn.execute("""
        INSERT INTO feedback_stock_gradient (
            ts_code, sample_count, avg_actual_pct_chg, hit_rate_limit,
            hit_rate_bigrise, avg_score, score_adjustment, updated_at
        )
        SELECT
            ts_code,
            COUNT(1) AS sample_count,
            AVG(actual_pct_chg) AS avg_actual_pct_chg,
            AVG(actual_limit) AS hit_rate_limit,
            AVG(actual_bigrise) AS hit_rate_bigrise,
            AVG(final_score) AS avg_score,
            MAX(-0.08, MIN(0.08,
                0.45 * (AVG(actual_bigrise) - 0.18)
                + 0.35 * (AVG(actual_limit) - 0.03)
                + 0.20 * (AVG(actual_pct_chg) / 10.0 - AVG(final_score))
            )) AS score_adjustment,
            datetime('now', 'localtime')
        FROM realized_predictions
        GROUP BY ts_code
        HAVING COUNT(1) >= 1
    """)

    conn.execute("DELETE FROM feedback_industry_gradient")
    conn.execute("""
        INSERT INTO feedback_industry_gradient (
            industry, sample_count, avg_actual_pct_chg, hit_rate_limit,
            hit_rate_bigrise, avg_score, score_adjustment, updated_at
        )
        SELECT
            COALESCE(NULLIF(TRIM(industry), ''), '未知') AS industry,
            COUNT(1) AS sample_count,
            AVG(actual_pct_chg) AS avg_actual_pct_chg,
            AVG(actual_limit) AS hit_rate_limit,
            AVG(actual_bigrise) AS hit_rate_bigrise,
            AVG(final_score) AS avg_score,
            MAX(-0.06, MIN(0.06,
                0.45 * (AVG(actual_bigrise) - 0.18)
                + 0.35 * (AVG(actual_limit) - 0.03)
                + 0.20 * (AVG(actual_pct_chg) / 10.0 - AVG(final_score))
            )) AS score_adjustment,
            datetime('now', 'localtime')
        FROM realized_predictions
        GROUP BY COALESCE(NULLIF(TRIM(industry), ''), '未知')
        HAVING COUNT(1) >= 3
    """)

    conn.commit()
    after = q(conn, "SELECT COUNT(1) FROM realized_predictions")[0][0]
    stock_n = q(conn, "SELECT COUNT(1) FROM feedback_stock_gradient")[0][0]
    industry_n = q(conn, "SELECT COUNT(1) FROM feedback_industry_gradient")[0][0]
    top_stock = q(conn, """
        SELECT ts_code, sample_count, ROUND(avg_actual_pct_chg, 3), ROUND(score_adjustment, 4)
        FROM feedback_stock_gradient
        ORDER BY score_adjustment DESC, sample_count DESC
        LIMIT 10
    """)
    top_industry = q(conn, """
        SELECT industry, sample_count, ROUND(avg_actual_pct_chg, 3), ROUND(score_adjustment, 4)
        FROM feedback_industry_gradient
        ORDER BY score_adjustment DESC, sample_count DESC
        LIMIT 10
    """)
    conn.close()

    print("\n修复结果")
    print(f"- 修复后 realized_predictions: {after}，新增/覆盖: {after - before}")
    print(f"- 股票梯度: {stock_n}")
    print(f"- 行业梯度: {industry_n}")
    print("\n股票正向梯度前十：")
    for row in top_stock:
        print(f"- {row[0]} 样本={row[1]} 实际均涨跌={row[2]} 调整={row[3]}")
    print("\n行业正向梯度前十：")
    for row in top_industry:
        print(f"- {row[0]} 样本={row[1]} 实际均涨跌={row[2]} 调整={row[3]}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--buffer-db", default=DEFAULT_BUFFER_DB)
    parser.add_argument("--data-db", default=DEFAULT_DATA_DB)
    parser.add_argument("--force-rebuild", action="store_true", help="备份并重建 realized/gradient 表")
    args = parser.parse_args()
    repair(args.buffer_db, args.data_db, args.force_rebuild)


if __name__ == "__main__":
    main()
