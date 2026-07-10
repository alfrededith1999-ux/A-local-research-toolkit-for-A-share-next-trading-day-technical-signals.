# -*- coding: utf-8 -*-
"""
TuShare 数据采集桌面应用 GUI
============================================================
功能：
1. 窗口输入 TuShare Token；
2. 勾选要下载的数据模块；
3. 进度条显示总体进度；
4. 日志窗口显示运行过程；
5. 报错时在窗口中显示错误；
6. 预留“运算脚本接口”，可接后续机器学习/特征工程/回测脚本；
7. 新增“只补缺失”模式：跳过已有基础表全量刷新，并把已有日期/窗口/区间数据自动登记为 done，避免重复拉取。

使用方式：
    1) 将本文件放到 .\tushare_gui_app.py
    2) 确保同目录下存在：tushare_permission_aligned_engine.py
    3) CMD 运行：
       cd /d .
       set PYTHONUTF8=1
       set PYTHONIOENCODING=utf-8
       python tushare_gui_app.py

依赖：
    python -m pip install -U tushare pandas tqdm

说明：
    GUI 本身不直接重写全部采集逻辑，而是动态加载 tushare_permission_aligned_engine.py。
    这样可以保持你原有数据库结构、补缺逻辑、落表命名和后续维护的一致性。
"""

from __future__ import annotations

import os
import sys
import time
import queue
import types
import traceback
import threading
import subprocess
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

# 尽量避免 Windows CMD GBK 输出导致 emoji/中文报错。
# 即使底层采集引擎里还有 emoji，这里也会把 stdout/stderr 重定向到 GUI 日志框。
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import tkinter as tk
from tkinter import ttk, filedialog, messagebox


# =============================================================================
# 1. 基础配置
# =============================================================================

APP_TITLE = "TuShare 数据采集控制台"
DEFAULT_DATA_DIR = r"."
DEFAULT_ENGINE_PATH = r"tushare_permission_aligned_engine.py"
DEFAULT_DB_PATH = r"data\tu_share_data.db"


@dataclass
class TaskSpec:
    key: str
    label: str
    func_name: str
    group: str
    default: bool = True
    heavy: bool = False
    note: str = ""


# 这里的 key 与 tushare_permission_aligned_engine.py 里的 ENABLE 字典保持一致。
# func_name 是对应的采集函数名。
TASKS: List[TaskSpec] = [
    TaskSpec("permission_inventory", "权限清单入库", "save_permission_inventory", "00. 权限与基础", True, False),
    TaskSpec("trade_calendar", "交易日历", "refresh_trade_calendar", "00. 权限与基础", True, False),
    TaskSpec("stock_basic", "A股基础表", "refresh_stock_basic", "00. 权限与基础", True, False),
    TaskSpec("etf_basic", "ETF基础表", "refresh_etf_basic", "00. 权限与基础", True, False),
    TaskSpec("sw_index_list", "申万指数列表", "refresh_sw_index_list", "00. 权限与基础", True, False),
    TaskSpec("hk_basic", "港股基础表", "refresh_hk_basic", "00. 权限与基础", True, False),
    TaskSpec("us_basic", "美股基础表", "refresh_us_basic", "00. 权限与基础", True, False),
    TaskSpec("fut_basic", "期货基础表", "refresh_fut_basic", "00. 权限与基础", True, False),
    TaskSpec("opt_basic", "期权基础表", "refresh_opt_basic", "00. 权限与基础", False, False),
    TaskSpec("cb_basic", "可转债基础表", "refresh_cb_basic", "00. 权限与基础", True, False),

    TaskSpec("daily_block", "A股日线/每日指标/资金/龙虎榜/大宗/涨跌停", "fetch_daily_block", "01. A股日线与专题", True, False),
    TaskSpec("premarket", "盘前股本", "fetch_premarket", "01. A股日线与专题", True, False),
    TaskSpec("auction", "集合竞价", "fetch_auction", "01. A股日线与专题", True, False),
    TaskSpec("irm_qa", "沪深董秘问答", "fetch_irm_qa", "01. A股日线与专题", True, False),
    TaskSpec("anns_d", "公告数据", "fetch_anns_d", "01. A股日线与专题", True, False),
    TaskSpec("news", "新闻资讯", "fetch_news", "01. A股日线与专题", True, False),
    TaskSpec("research_report", "券商研报", "fetch_research_report", "01. A股日线与专题", True, False),
    TaskSpec("policy_npr", "政策法规库", "fetch_policy_npr", "01. A股日线与专题", True, False),
    TaskSpec("cb_price_chg", "可转债价格变动", "fetch_cb_price_chg", "01. A股日线与专题", True, False),

    TaskSpec("stk_mins", "A股历史分钟", "fetch_stk_mins", "02. 历史分钟", False, True, "体量很大，建议先小范围测试"),
    TaskSpec("etf_mins", "ETF历史分钟", "fetch_etf_mins", "02. 历史分钟", False, True, "体量很大，建议先小范围测试"),
    TaskSpec("sw_mins", "申万指数分钟", "fetch_sw_mins", "02. 历史分钟", False, True, "需要申万指数列表"),
    TaskSpec("fut_mins", "期货历史分钟", "fetch_fut_mins", "02. 历史分钟", False, True),
    TaskSpec("opt_mins", "期权历史分钟", "fetch_opt_mins", "02. 历史分钟", False, True),
    TaskSpec("hk_mins", "港股历史分钟", "fetch_hk_mins", "02. 历史分钟", False, True),

    TaskSpec("hk_daily", "港股历史日线", "fetch_hk_daily", "03. 港美股与财报", True, False),
    TaskSpec("us_daily", "美股历史日线", "fetch_us_daily", "03. 港美股与财报", True, False),
    TaskSpec("us_financials", "美股财报", "fetch_us_financials", "03. 港美股与财报", False, True),
    TaskSpec("hk_financials", "港股财报", "fetch_hk_financials", "03. 港美股与财报", False, True),

    TaskSpec("rt_a_stock_min", "A股分钟RT", "fetch_rt_a_stock_min", "04. 实时快照", True, False),
    TaskSpec("rt_etf_min", "ETF分钟RT", "fetch_rt_etf_min", "04. 实时快照", True, False),
    TaskSpec("rt_index_min", "指数分钟RT", "fetch_rt_index_min", "04. 实时快照", True, False),
    TaskSpec("rt_fut_min", "期货分钟RT", "fetch_rt_fut_min", "04. 实时快照", False, False),
    TaskSpec("rt_a_stock_k", "A股日线RT", "fetch_rt_a_stock_k", "04. 实时快照", True, False),
    TaskSpec("rt_etf_k", "ETF日线RT", "fetch_rt_etf_k", "04. 实时快照", True, False),
    TaskSpec("rt_index_k", "指数日线RT", "fetch_rt_index_k", "04. 实时快照", True, False),
    TaskSpec("rt_hk_k", "港股实时日线", "fetch_rt_hk_k", "04. 实时快照", True, False),
]


# =============================================================================
# 2. GUI 日志重定向
# =============================================================================

class QueueWriter:
    """把 print/tqdm 输出写入线程安全队列，再由 GUI 主线程刷新到日志框。"""

    def __init__(self, log_queue: "queue.Queue[tuple]", stream_name: str = "stdout"):
        self.log_queue = log_queue
        self.stream_name = stream_name
        self._buf = ""

    def write(self, text: str) -> int:
        if not text:
            return 0
        text = text.replace("\r", "\n")
        self._buf += text
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if line.strip():
                self.log_queue.put(("log", line))
        return len(text)

    def flush(self) -> None:
        if self._buf.strip():
            self.log_queue.put(("log", self._buf.strip()))
        self._buf = ""


class StdRedirect:
    """上下文管理器：临时把 stdout/stderr 指向 QueueWriter。"""

    def __init__(self, writer: QueueWriter):
        self.writer = writer
        self.old_stdout = None
        self.old_stderr = None

    def __enter__(self):
        self.old_stdout = sys.stdout
        self.old_stderr = sys.stderr
        sys.stdout = self.writer  # type: ignore[assignment]
        sys.stderr = self.writer  # type: ignore[assignment]
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            self.writer.flush()
        finally:
            if self.old_stdout is not None:
                sys.stdout = self.old_stdout
            if self.old_stderr is not None:
                sys.stderr = self.old_stderr


# =============================================================================
# 3. 采集运行器
# =============================================================================

class TushareGuiRunner:
    def __init__(
        self,
        token: str,
        db_path: str,
        engine_path: str,
        selected_keys: List[str],
        log_queue: "queue.Queue[tuple]",
        stop_event: threading.Event,
        only_missing: bool = True,
    ):
        self.token = token.strip()
        self.db_path = db_path.strip()
        self.engine_path = engine_path.strip()
        self.selected_keys = selected_keys
        self.log_queue = log_queue
        self.stop_event = stop_event
        self.only_missing = bool(only_missing)
        self.engine: Optional[types.ModuleType] = None
        self.conn = None
        self.sw_codes: List[str] = []
        self._table_cols_cache: Dict[str, List[str]] = {}
        self._table_exists_cache: Dict[str, bool] = {}

    def log(self, text: str) -> None:
        self.log_queue.put(("log", text))

    def progress(self, current: int, total: int, label: str) -> None:
        self.log_queue.put(("progress", current, total, label))

    def _load_engine(self) -> types.ModuleType:
        if not self.token:
            raise RuntimeError("请先输入 TuShare Token。")
        if not Path(self.engine_path).exists():
            raise FileNotFoundError(f"未找到采集引擎文件：{self.engine_path}")

        os.environ["TUSHARE_TOKEN"] = self.token
        os.environ.setdefault("PYTHONUTF8", "1")
        os.environ.setdefault("PYTHONIOENCODING", "utf-8")

        module_name = f"tushare_engine_runtime_{int(time.time() * 1000)}"
        spec = importlib.util.spec_from_file_location(module_name, self.engine_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"无法加载采集引擎：{self.engine_path}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        # 覆盖路径配置，确保写入 GUI 指定数据库。
        data_dir = str(Path(self.db_path).parent)
        Path(data_dir).mkdir(parents=True, exist_ok=True)
        module.DATA_DIR = data_dir
        module.DB_PATH = self.db_path

        # 覆盖 Token/pro 对象，防止用户 GUI 内临时换 token。
        try:
            module.TS_TOKEN = self.token
            module.ts.set_token(self.token)
            module.pro = module.ts.pro_api(self.token)
        except Exception as e:
            raise RuntimeError(f"Token 初始化失败：{e}") from e

        # GUI 负责决定哪些任务启用。
        if hasattr(module, "ENABLE") and isinstance(module.ENABLE, dict):
            for k in list(module.ENABLE.keys()):
                module.ENABLE[k] = k in self.selected_keys
            # 永远禁止交易所指数历史分钟，避免误触 idx_mins 权限。
            module.ENABLE["exchange_idx_mins_history"] = False

        return module

    def _call(self, func_name: str, *args):
        if self.engine is None:
            raise RuntimeError("采集引擎尚未加载。")
        func = getattr(self.engine, func_name, None)
        if func is None:
            self.log(f"[SKIP] 引擎中没有函数：{func_name}")
            return None
        return func(*args)



    # ------------------------------------------------------------------
    # 只补缺失模式：运行时给采集引擎打补丁
    # ------------------------------------------------------------------

    def _table_exists(self, table: str) -> bool:
        if self.engine is None or self.conn is None:
            return False
        if table in self._table_exists_cache:
            return self._table_exists_cache[table]
        try:
            ok = bool(self.engine.table_exists(self.conn, table))
        except Exception:
            ok = False
        self._table_exists_cache[table] = ok
        return ok

    def _table_cols(self, table: str) -> List[str]:
        if self.engine is None or self.conn is None or not self._table_exists(table):
            return []
        if table in self._table_cols_cache:
            return self._table_cols_cache[table]
        try:
            cols = list(self.engine.get_table_cols(self.conn, table))
        except Exception:
            try:
                cur = self.conn.cursor()
                cur.execute(f'PRAGMA table_info("{table}");')
                cols = [r[1] for r in cur.fetchall()]
            except Exception:
                cols = []
        self._table_cols_cache[table] = cols
        return cols

    def _count_rows(self, table: str, where_sql: str = "", params: tuple = ()) -> int:
        if self.engine is None or self.conn is None or not self._table_exists(table):
            return 0
        try:
            sql = f'SELECT COUNT(1) AS n FROM "{table}"'
            if where_sql:
                sql += f' WHERE {where_sql}'
            cur = self.conn.cursor()
            cur.execute(sql, params)
            row = cur.fetchone()
            return int(row[0] or 0) if row else 0
        except Exception:
            return 0

    def _optimize_sqlite_connection(self) -> None:
        if self.conn is None:
            return
        pragmas = [
            "PRAGMA journal_mode=WAL;",
            "PRAGMA synchronous=NORMAL;",
            "PRAGMA temp_store=MEMORY;",
            "PRAGMA cache_size=-300000;",
            "PRAGMA busy_timeout=120000;",
        ]
        for sql in pragmas:
            try:
                self.conn.execute(sql)
            except Exception:
                pass

    def _ensure_done_indexes(self) -> None:
        if self.conn is None:
            return
        stmts = [
            'CREATE INDEX IF NOT EXISTS idx_window_done_fast ON window_done(dataset, object_key, start_ymd, end_ymd);',
            'CREATE INDEX IF NOT EXISTS idx_object_done_fast ON object_done(dataset, object_key, start_ymd, end_ymd);',
        ]
        for sql in stmts:
            try:
                self.conn.execute(sql)
            except Exception:
                pass
        try:
            self.conn.commit()
        except Exception:
            pass

    def _table_has_any(self, table: str) -> bool:
        return self._count_rows(table) > 0

    def _date_range_has_rows(self, table: str, date_col: str, ws: str, we: str) -> bool:
        cols = self._table_cols(table)
        if date_col in cols:
            return self._count_rows(table, f'"{date_col}">=? AND "{date_col}"<=?', (ws, we)) > 0
        # 兜底：常见日期列自动探测。
        for c in ["trade_date", "ann_date", "pub_date", "date", "cal_date", "end_date", "report_date", "f_ann_date", "datetime", "pub_time"]:
            if c in cols:
                try:
                    return self._count_rows(table, f'substr("{c}",1,8)>=? AND substr("{c}",1,8)<=?', (ws, we)) > 0
                except Exception:
                    pass
        return self._table_has_any(table)

    def _mins_window_has_rows(self, table: str, ts_code: str, ws: str, we: str) -> bool:
        if self.engine is None:
            return False
        cols = self._table_cols(table)
        if not cols:
            return False
        code_filter = 'ts_code=? AND ' if "ts_code" in cols else ""
        code_params: tuple = (ts_code,) if "ts_code" in cols else ()
        if "trade_time" in cols:
            s = self.engine.ymd_to_dt(ws, "00:00:00")
            e = self.engine.ymd_to_dt(we, "23:59:59")
            return self._count_rows(table, f'{code_filter}"trade_time">=? AND "trade_time"<=?', code_params + (s, e)) > 0
        if "trade_date" in cols:
            return self._count_rows(table, f'{code_filter}"trade_date">=? AND "trade_date"<=?', code_params + (ws, we)) > 0
        return False

    def _read_sw_codes_from_table(self) -> List[str]:
        if self.engine is None or self.conn is None or not self._table_has_any("sw_index_list"):
            return []
        cols = self._table_cols("sw_index_list")
        code_col = "ts_code" if "ts_code" in cols else ("index_code" if "index_code" in cols else "")
        if not code_col:
            return []
        try:
            df = self.engine.q(self.conn, f'SELECT DISTINCT "{code_col}" AS code FROM "sw_index_list" WHERE "{code_col}" IS NOT NULL;')
            out = []
            for c in df["code"].astype(str).tolist():
                c = c.strip()
                if not c:
                    continue
                if c.endswith(".SI"):
                    out.append(c)
                elif c.isdigit():
                    out.append(c + ".SI")
                else:
                    out.append(c if "." in c else c + ".SI")
            return sorted(set(out))
        except Exception:
            return []

    def _install_only_missing_patches(self) -> None:
        """在不改动采集引擎源码的前提下，让 GUI 运行时强制采用“只补缺失”。

        作用：
        1) 基础全量表如果已有数据，则跳过 DELETE + 全表刷新；
        2) 交易日历只追加缺失日期，不覆盖旧表；
        3) 历史分钟先扫描已有窗口并登记 window_done，再只拉缺失窗口；
        4) range 接口先扫描已有区间并登记 object_done，再只拉缺失区间；
        5) 财报等按代码接口会尽量用已有 ts_code 记录推断 object_done。
        """
        if self.engine is None or self.conn is None:
            return
        m = self.engine
        self.log("[MISSING-ONLY] 已启用只补缺失模式：不会主动全量覆盖已有表。")

        # ---- 1) 权限清单：已有就跳过 ----
        if hasattr(m, "save_permission_inventory"):
            original_save_permission_inventory = m.save_permission_inventory

            def save_permission_inventory_missing_only(conn):
                if self._table_has_any("tushare_permission_inventory"):
                    self.log("[MISSING-ONLY] tushare_permission_inventory 已存在，跳过权限清单重写。")
                    return None
                return original_save_permission_inventory(conn)

            m.save_permission_inventory = save_permission_inventory_missing_only

        # ---- 2) 交易日历：只追加 max(cal_date) 之后的日期 ----
        if hasattr(m, "refresh_trade_calendar"):
            original_refresh_trade_calendar = m.refresh_trade_calendar

            def refresh_trade_calendar_missing_only(conn):
                if not self._table_has_any("trade_calendar"):
                    self.log("[MISSING-ONLY] trade_calendar 为空，执行首次初始化。")
                    return original_refresh_trade_calendar(conn)
                try:
                    df_max = m.q(conn, 'SELECT MAX(cal_date) AS d FROM "trade_calendar";')
                    max_d = str(df_max.loc[0, "d"]) if not df_max.empty and df_max.loc[0, "d"] is not None else ""
                    today = m.today_cn_ymd()
                    if max_d >= today:
                        self.log(f"[MISSING-ONLY] trade_calendar 已覆盖到 {max_d}，跳过刷新。")
                        return None
                    start_dt = (m.datetime.datetime.strptime(max_d, "%Y%m%d") + m.datetime.timedelta(days=1)).strftime("%Y%m%d")
                    self.log(f"[MISSING-ONLY] trade_calendar 仅追加 {start_dt} - {today}。")
                    df = m.safe_call("standard", m.pro.trade_cal, exchange="", start_date=start_dt, end_date=today,
                                     fields="exchange,cal_date,is_open,pretrade_date")
                    if df is None or df.empty:
                        self.log("[MISSING-ONLY] trade_calendar 无新增日期。")
                        return None
                    m.ensure_table_and_columns(conn, "trade_calendar", df)
                    m.insert_df_fast(conn, "trade_calendar", df)
                    return None
                except Exception:
                    self.log("[MISSING-ONLY] trade_calendar 增量追加失败，回退到原刷新函数。")
                    return original_refresh_trade_calendar(conn)

            m.refresh_trade_calendar = refresh_trade_calendar_missing_only

        # ---- 3) 基础全量表：已有就跳过，避免 DELETE + 全量重拉 ----
        full_refresh_map = {
            "refresh_stock_basic": "stock_basic",
            "refresh_etf_basic": "etf_basic",
            "refresh_hk_basic": "hk_basic",
            "refresh_us_basic": "us_basic",
            "refresh_fut_basic": "fut_basic",
            "refresh_opt_basic": "opt_basic",
            "refresh_cb_basic": "cb_basic",
        }
        for func_name, table in full_refresh_map.items():
            if not hasattr(m, func_name):
                continue
            original = getattr(m, func_name)

            def make_wrapper(orig, tbl, fname):
                def wrapper(conn):
                    if self._table_has_any(tbl):
                        self.log(f"[MISSING-ONLY] {tbl} 已存在数据，跳过 {fname} 全量刷新。")
                        return None
                    self.log(f"[MISSING-ONLY] {tbl} 不存在或为空，执行首次采集。")
                    return orig(conn)
                return wrapper

            setattr(m, func_name, make_wrapper(original, table, func_name))

        # ---- 4) 申万列表：已有就返回已有代码，避免 DELETE + 全量重拉 ----
        if hasattr(m, "refresh_sw_index_list"):
            original_refresh_sw_index_list = m.refresh_sw_index_list

            def refresh_sw_index_list_missing_only(conn):
                if self._table_has_any("sw_index_list"):
                    codes = self._read_sw_codes_from_table()
                    self.log(f"[MISSING-ONLY] sw_index_list 已存在，跳过全量刷新；复用 {len(codes)} 个申万代码。")
                    return codes
                return original_refresh_sw_index_list(conn)

            m.refresh_sw_index_list = refresh_sw_index_list_missing_only

        # ---- 5) 历史分钟：先把已有数据窗口登记为 window_done，再交给原函数补真正缺失窗口 ----
        if hasattr(m, "patch_mins_windows"):
            original_patch_mins_windows = m.patch_mins_windows

            def patch_mins_windows_missing_only(conn, dataset, table, api_name, codes, start_ymd, end_ymd, tag="minute",
                                                start_hhmmss="09:00:00", end_hhmmss="15:00:00", extra_kwargs=None):
                try:
                    if self._table_has_any(table):
                        wins = m.month_windows(start_ymd, end_ymd)
                        premarked = 0
                        for c in sorted(set([str(x) for x in codes if str(x).strip()])):
                            for ws, we in wins:
                                if m.window_is_done(conn, dataset, c, ws, we):
                                    continue
                                if self._mins_window_has_rows(table, c, ws, we):
                                    m.mark_window_done(conn, dataset, c, ws, we)
                                    premarked += 1
                        if premarked:
                            self.log(f"[MISSING-ONLY] {dataset} 已根据现有数据登记 {premarked} 个已有窗口，后续只补缺失窗口。")
                except Exception as e:
                    self.log(f"[MISSING-ONLY] {dataset} 预扫描已有窗口失败，将继续使用原补采逻辑：{e}")
                return original_patch_mins_windows(conn, dataset, table, api_name, codes, start_ymd, end_ymd, tag,
                                                   start_hhmmss, end_hhmmss, extra_kwargs)

            m.patch_mins_windows = patch_mins_windows_missing_only

        # ---- 6) range 表：先把已有区间登记为 object_done，再只补缺失区间 ----
        if hasattr(m, "patch_range_table"):
            original_patch_range_table = m.patch_range_table

            def patch_range_table_missing_only(conn, dataset, table, api_name, start_ymd, end_ymd, tag, date_col,
                                               start_param="start_date", end_param="end_date", step_days=30, extra_kwargs=None):
                try:
                    if self._table_has_any(table):
                        wins = m.day_windows(start_ymd, end_ymd, step_days=step_days)
                        premarked = 0
                        for ws, we in wins:
                            key = f"{ws}_{we}"
                            if m.object_is_done(conn, dataset, key, ws, we):
                                continue
                            if self._date_range_has_rows(table, date_col, ws, we):
                                m.mark_object_done(conn, dataset, key, ws, we)
                                premarked += 1
                        if premarked:
                            self.log(f"[MISSING-ONLY] {dataset} 已根据现有数据登记 {premarked} 个已有区间，后续只补缺失区间。")
                except Exception as e:
                    self.log(f"[MISSING-ONLY] {dataset} 预扫描已有区间失败，将继续使用原补采逻辑：{e}")
                return original_patch_range_table(conn, dataset, table, api_name, start_ymd, end_ymd, tag, date_col,
                                                  start_param, end_param, step_days, extra_kwargs)

            m.patch_range_table = patch_range_table_missing_only

        # ---- 7) 财报等按代码任务：object_done 缺失时，尽量从已有 ts_code 记录推断 ----
        if hasattr(m, "object_is_done") and hasattr(m, "mark_object_done"):
            original_object_is_done = m.object_is_done

            def object_is_done_missing_only(conn, dataset, object_key, ws, we):
                try:
                    if original_object_is_done(conn, dataset, object_key, ws, we):
                        return True
                    table = dataset
                    if not self._table_has_any(table):
                        return False
                    cols = self._table_cols(table)
                    # 批量 key、ALL 这类对象不做自动跳过，避免误判。
                    if not object_key or object_key in {"ALL", "all"} or "," in str(object_key):
                        return False
                    if "ts_code" in cols:
                        if ws != "ALL" and we != "ALL":
                            # 常见财报/公告日期列兜底。
                            for dc in ["ann_date", "f_ann_date", "end_date", "report_date", "trade_date", "pub_date"]:
                                if dc in cols:
                                    if self._count_rows(table, 'ts_code=? AND substr("{}",1,8)>=? AND substr("{}",1,8)<=?'.format(dc, dc), (object_key, ws, we)) > 0:
                                        m.mark_object_done(conn, dataset, object_key, ws, we)
                                        return True
                            if self._count_rows(table, 'ts_code=?', (object_key,)) > 0:
                                m.mark_object_done(conn, dataset, object_key, ws, we)
                                return True
                        else:
                            if self._count_rows(table, 'ts_code=?', (object_key,)) > 0:
                                m.mark_object_done(conn, dataset, object_key, ws, we)
                                return True
                    return False
                except Exception:
                    return original_object_is_done(conn, dataset, object_key, ws, we)

            m.object_is_done = object_is_done_missing_only

    def _needs_trade_calendar(self) -> bool:
        date_based = {
            "daily_block", "premarket", "auction", "irm_qa", "anns_d", "cb_price_chg",
            "stk_mins", "etf_mins", "sw_mins", "fut_mins", "opt_mins", "hk_mins",
        }
        return any(k in self.selected_keys for k in date_based)

    def _needs_stock_basic(self) -> bool:
        return any(k in self.selected_keys for k in ["stk_mins", "rt_a_stock_min", "rt_a_stock_k"])

    def _needs_etf_basic(self) -> bool:
        return any(k in self.selected_keys for k in ["etf_mins", "rt_etf_min", "rt_etf_k"])

    def _needs_sw_list(self) -> bool:
        return "sw_mins" in self.selected_keys

    def run(self) -> None:
        writer = QueueWriter(self.log_queue)
        with StdRedirect(writer):
            try:
                self.log("========== TuShare GUI 采集任务开始 ==========")
                self.log(f"数据库：{self.db_path}")
                self.log(f"采集引擎：{self.engine_path}")
                self.log(f"已选择模块数：{len(self.selected_keys)}")

                self.engine = self._load_engine()
                self.conn = self.engine.connect_db()
                self._optimize_sqlite_connection()

                # 确保 done 表存在。
                if hasattr(self.engine, "ensure_done_tables"):
                    self.engine.ensure_done_tables(self.conn)
                self._ensure_done_indexes()

                if self.only_missing:
                    self._install_only_missing_patches()
                else:
                    self.log("[MODE] 未启用只补缺失：将按采集引擎原始逻辑运行，可能覆盖基础表。")

                # 自动补充必要前置数据，不把它们算入用户勾选也可以。
                if self._needs_trade_calendar() and "trade_calendar" not in self.selected_keys:
                    self.log("[AUTO] 当前任务需要交易日历，自动刷新 trade_calendar。")
                    self.engine.refresh_trade_calendar(self.conn)
                if self._needs_stock_basic() and "stock_basic" not in self.selected_keys:
                    self.log("[AUTO] 当前任务需要 A股基础表，自动刷新 stock_basic。")
                    self.engine.refresh_stock_basic(self.conn)
                if self._needs_etf_basic() and "etf_basic" not in self.selected_keys:
                    self.log("[AUTO] 当前任务需要 ETF基础表，自动刷新 etf_basic。")
                    self.engine.refresh_etf_basic(self.conn)
                if self._needs_sw_list() and "sw_index_list" not in self.selected_keys:
                    self.log("[AUTO] 当前任务需要申万指数列表，自动刷新 sw_index_list。")
                    self.sw_codes = self.engine.refresh_sw_index_list(self.conn) or []

                selected_tasks = [t for t in TASKS if t.key in self.selected_keys]
                total = len(selected_tasks)
                done = 0
                self.progress(done, total, "准备开始")

                for task in selected_tasks:
                    if self.stop_event.is_set():
                        self.log("[STOP] 用户请求停止。当前模块结束后已停止继续运行。")
                        break

                    label = f"{task.group} - {task.label}"
                    self.log(f"\n---------- 开始：{label} ----------")
                    self.progress(done, total, label)

                    try:
                        if task.key == "permission_inventory":
                            self.engine.save_permission_inventory(self.conn)
                        elif task.key == "sw_index_list":
                            self.sw_codes = self.engine.refresh_sw_index_list(self.conn) or []
                        elif task.key == "sw_mins":
                            if not self.sw_codes:
                                self.sw_codes = self.engine.refresh_sw_index_list(self.conn) or []
                            self.engine.fetch_sw_mins(self.conn, self.sw_codes)
                        else:
                            self._call(task.func_name, self.conn)
                        self.log(f"[OK] 完成：{label}")
                    except Exception:
                        err = traceback.format_exc()
                        self.log(f"[ERROR] 模块失败：{label}\n{err}")
                        self.log_queue.put(("error", f"模块失败：{label}", err))
                        # 出错后不继续跑，避免数据库处于不确定状态。
                        raise
                    finally:
                        done += 1
                        self.progress(done, total, label)

                try:
                    if self.conn is not None:
                        self.conn.close()
                except Exception:
                    pass

                self.log("========== TuShare GUI 采集任务结束 ==========")
                self.log_queue.put(("done", "采集任务已完成"))
            except Exception:
                err = traceback.format_exc()
                self.log(f"[FATAL] 任务终止：\n{err}")
                self.log_queue.put(("fatal", "采集任务失败", err))


# =============================================================================
# 4. 主窗口
# =============================================================================

class TushareGuiApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1180x780")
        self.minsize(1060, 700)

        self.log_queue: "queue.Queue[tuple]" = queue.Queue()
        self.stop_event = threading.Event()
        self.worker_thread: Optional[threading.Thread] = None
        self.analysis_thread: Optional[threading.Thread] = None

        self.var_token = tk.StringVar(value=os.getenv("TUSHARE_TOKEN", ""))
        self.var_show_token = tk.BooleanVar(value=False)
        self.var_engine_path = tk.StringVar(value=DEFAULT_ENGINE_PATH)
        self.var_db_path = tk.StringVar(value=DEFAULT_DB_PATH)
        self.var_analysis_script = tk.StringVar(value="")
        self.var_save_token_env = tk.BooleanVar(value=False)
        self.var_only_missing = tk.BooleanVar(value=True)

        self.task_vars: Dict[str, tk.BooleanVar] = {
            t.key: tk.BooleanVar(value=t.default) for t in TASKS
        }

        self._build_ui()
        self._poll_queue()

    # ------------------------------------------------------------------
    # UI 构建
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(3, weight=1)

        top = ttk.LabelFrame(self, text="1. Token 与路径配置")
        top.grid(row=0, column=0, padx=10, pady=(10, 6), sticky="ew")
        top.columnconfigure(1, weight=1)

        ttk.Label(top, text="TuShare Token：").grid(row=0, column=0, padx=8, pady=6, sticky="w")
        self.entry_token = ttk.Entry(top, textvariable=self.var_token, show="*")
        self.entry_token.grid(row=0, column=1, padx=8, pady=6, sticky="ew")
        ttk.Checkbutton(top, text="显示", variable=self.var_show_token, command=self._toggle_token_show).grid(row=0, column=2, padx=4, pady=6)
        ttk.Checkbutton(top, text="保存到环境变量", variable=self.var_save_token_env).grid(row=0, column=3, padx=4, pady=6)
        ttk.Button(top, text="测试Token", command=self.test_token).grid(row=0, column=4, padx=8, pady=6)

        ttk.Label(top, text="采集引擎：").grid(row=1, column=0, padx=8, pady=6, sticky="w")
        ttk.Entry(top, textvariable=self.var_engine_path).grid(row=1, column=1, columnspan=3, padx=8, pady=6, sticky="ew")
        ttk.Button(top, text="选择", command=self.choose_engine).grid(row=1, column=4, padx=8, pady=6)

        ttk.Label(top, text="数据库：").grid(row=2, column=0, padx=8, pady=6, sticky="w")
        ttk.Entry(top, textvariable=self.var_db_path).grid(row=2, column=1, columnspan=3, padx=8, pady=6, sticky="ew")
        ttk.Button(top, text="选择", command=self.choose_db).grid(row=2, column=4, padx=8, pady=6)

        controls = ttk.LabelFrame(self, text="2. 选择要下载的数据")
        controls.grid(row=1, column=0, padx=10, pady=6, sticky="ew")
        controls.columnconfigure(0, weight=1)

        button_bar = ttk.Frame(controls)
        button_bar.grid(row=0, column=0, padx=8, pady=(8, 4), sticky="ew")
        ttk.Button(button_bar, text="勾选默认推荐", command=self.select_default).pack(side="left", padx=(0, 6))
        ttk.Button(button_bar, text="勾选全部权限", command=self.select_all).pack(side="left", padx=6)
        ttk.Button(button_bar, text="只勾选基础+日线", command=self.select_basic_daily).pack(side="left", padx=6)
        ttk.Button(button_bar, text="清空", command=self.select_none).pack(side="left", padx=6)
        ttk.Checkbutton(button_bar, text="只补缺失（不全量重拉）", variable=self.var_only_missing).pack(side="left", padx=12)
        ttk.Label(button_bar, text="提示：历史分钟和财报体量较大，建议先小范围跑。", foreground="#666666").pack(side="left", padx=16)

        self.task_frame_container = ttk.Frame(controls)
        self.task_frame_container.grid(row=1, column=0, padx=8, pady=(2, 8), sticky="ew")
        self._build_task_checkboxes(self.task_frame_container)

        action = ttk.LabelFrame(self, text="3. 运行控制")
        action.grid(row=2, column=0, padx=10, pady=6, sticky="ew")
        action.columnconfigure(3, weight=1)

        self.btn_start = ttk.Button(action, text="开始下载", command=self.start_download)
        self.btn_start.grid(row=0, column=0, padx=8, pady=8)
        self.btn_stop = ttk.Button(action, text="停止", command=self.stop_download, state="disabled")
        self.btn_stop.grid(row=0, column=1, padx=8, pady=8)
        ttk.Button(action, text="打开数据库目录", command=self.open_db_folder).grid(row=0, column=2, padx=8, pady=8)

        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress_bar = ttk.Progressbar(action, variable=self.progress_var, maximum=100.0)
        self.progress_bar.grid(row=0, column=3, padx=8, pady=8, sticky="ew")
        self.status_label = ttk.Label(action, text="未开始")
        self.status_label.grid(row=0, column=4, padx=8, pady=8, sticky="e")

        analysis = ttk.LabelFrame(self, text="4. 预留运算脚本接口")
        analysis.grid(row=3, column=0, padx=10, pady=6, sticky="ew")
        analysis.columnconfigure(1, weight=1)
        ttk.Label(analysis, text="运算脚本：").grid(row=0, column=0, padx=8, pady=6, sticky="w")
        ttk.Entry(analysis, textvariable=self.var_analysis_script).grid(row=0, column=1, padx=8, pady=6, sticky="ew")
        ttk.Button(analysis, text="选择", command=self.choose_analysis_script).grid(row=0, column=2, padx=8, pady=6)
        ttk.Button(analysis, text="运行运算脚本", command=self.run_analysis_script).grid(row=0, column=3, padx=8, pady=6)
        ttk.Label(
            analysis,
            text="默认会传入参数：--db <数据库路径> --data-dir <数据库目录>，并设置环境变量 TUSHARE_DB_PATH。",
            foreground="#666666",
        ).grid(row=1, column=0, columnspan=4, padx=8, pady=(0, 6), sticky="w")

        log_box = ttk.LabelFrame(self, text="5. 日志与错误信息")
        log_box.grid(row=4, column=0, padx=10, pady=(6, 10), sticky="nsew")
        self.rowconfigure(4, weight=1)
        log_box.columnconfigure(0, weight=1)
        log_box.rowconfigure(0, weight=1)

        self.text_log = tk.Text(log_box, wrap="word", height=16, font=("Consolas", 10))
        self.text_log.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(log_box, orient="vertical", command=self.text_log.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.text_log.configure(yscrollcommand=scroll.set)

        log_button_bar = ttk.Frame(log_box)
        log_button_bar.grid(row=1, column=0, columnspan=2, sticky="ew")
        ttk.Button(log_button_bar, text="清空日志", command=self.clear_log).pack(side="left", padx=6, pady=5)
        ttk.Button(log_button_bar, text="保存日志", command=self.save_log).pack(side="left", padx=6, pady=5)

    def _build_task_checkboxes(self, parent: ttk.Frame) -> None:
        # 分组横向排列，避免窗口过长。
        groups: Dict[str, List[TaskSpec]] = {}
        for task in TASKS:
            groups.setdefault(task.group, []).append(task)

        canvas = tk.Canvas(parent, height=220, highlightthickness=0)
        hbar = ttk.Scrollbar(parent, orient="horizontal", command=canvas.xview)
        vbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        inner = ttk.Frame(canvas)

        inner_id = canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(xscrollcommand=hbar.set, yscrollcommand=vbar.set)

        def on_configure(_event=None):
            canvas.configure(scrollregion=canvas.bbox("all"))

        inner.bind("<Configure>", on_configure)

        col = 0
        for group_name, tasks in groups.items():
            frame = ttk.LabelFrame(inner, text=group_name)
            frame.grid(row=0, column=col, padx=8, pady=6, sticky="n")
            for r, task in enumerate(tasks):
                label = task.label + ("  [大]" if task.heavy else "")
                cb = ttk.Checkbutton(frame, text=label, variable=self.task_vars[task.key])
                cb.grid(row=r, column=0, sticky="w", padx=6, pady=2)
            col += 1

        canvas.grid(row=0, column=0, sticky="ew")
        vbar.grid(row=0, column=1, sticky="ns")
        hbar.grid(row=1, column=0, sticky="ew")
        parent.columnconfigure(0, weight=1)

        # 鼠标滚轮支持。
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)

    # ------------------------------------------------------------------
    # UI 工具函数
    # ------------------------------------------------------------------

    def _toggle_token_show(self) -> None:
        self.entry_token.configure(show="" if self.var_show_token.get() else "*")

    def append_log(self, text: str) -> None:
        ts = time.strftime("%H:%M:%S")
        self.text_log.insert("end", f"[{ts}] {text}\n")
        self.text_log.see("end")

    def clear_log(self) -> None:
        self.text_log.delete("1.0", "end")

    def save_log(self) -> None:
        path = filedialog.asksaveasfilename(
            title="保存日志",
            defaultextension=".log",
            filetypes=[("Log files", "*.log"), ("Text files", "*.txt"), ("All files", "*.*")],
            initialdir=str(Path(self.var_db_path.get()).parent),
        )
        if not path:
            return
        Path(path).write_text(self.text_log.get("1.0", "end"), encoding="utf-8")
        messagebox.showinfo("已保存", f"日志已保存到：\n{path}")

    def choose_engine(self) -> None:
        path = filedialog.askopenfilename(
            title="选择采集引擎脚本",
            filetypes=[("Python files", "*.py"), ("All files", "*.*")],
            initialdir=str(Path(self.var_engine_path.get()).parent),
        )
        if path:
            self.var_engine_path.set(path)

    def choose_db(self) -> None:
        path = filedialog.asksaveasfilename(
            title="选择/创建数据库",
            defaultextension=".db",
            filetypes=[("SQLite DB", "*.db"), ("All files", "*.*")],
            initialdir=str(Path(self.var_db_path.get()).parent),
            initialfile=Path(self.var_db_path.get()).name,
        )
        if path:
            self.var_db_path.set(path)

    def choose_analysis_script(self) -> None:
        path = filedialog.askopenfilename(
            title="选择后续运算脚本",
            filetypes=[("Python files", "*.py"), ("All files", "*.*")],
            initialdir=str(Path(self.var_db_path.get()).parent),
        )
        if path:
            self.var_analysis_script.set(path)

    def open_db_folder(self) -> None:
        folder = str(Path(self.var_db_path.get()).parent)
        try:
            os.startfile(folder)  # type: ignore[attr-defined]
        except Exception as e:
            messagebox.showerror("打开失败", str(e))

    def select_default(self) -> None:
        for task in TASKS:
            self.task_vars[task.key].set(task.default)

    def select_all(self) -> None:
        for task in TASKS:
            self.task_vars[task.key].set(True)

    def select_none(self) -> None:
        for task in TASKS:
            self.task_vars[task.key].set(False)

    def select_basic_daily(self) -> None:
        allow_groups = {"00. 权限与基础", "01. A股日线与专题", "03. 港美股与财报", "04. 实时快照"}
        for task in TASKS:
            self.task_vars[task.key].set(task.group in allow_groups and not task.heavy)

    def get_selected_keys(self) -> List[str]:
        return [k for k, v in self.task_vars.items() if v.get()]

    def validate_inputs(self) -> bool:
        if not self.var_token.get().strip():
            messagebox.showwarning("缺少 Token", "请先输入 TuShare Token。")
            return False
        if not Path(self.var_engine_path.get()).exists():
            messagebox.showwarning("缺少采集引擎", f"找不到采集引擎：\n{self.var_engine_path.get()}")
            return False
        selected = self.get_selected_keys()
        if not selected:
            messagebox.showwarning("未选择数据", "请至少勾选一个要下载的数据模块。")
            return False
        heavy_selected = [t.label for t in TASKS if t.key in selected and t.heavy]
        if heavy_selected:
            msg = "你勾选了体量较大的模块：\n\n" + "\n".join(heavy_selected[:12])
            if len(heavy_selected) > 12:
                msg += f"\n……等 {len(heavy_selected)} 项"
            msg += "\n\n这些模块可能运行很久。是否继续？"
            if not messagebox.askyesno("确认运行大体量模块", msg):
                return False
        return True

    # ------------------------------------------------------------------
    # Token 测试
    # ------------------------------------------------------------------

    def test_token(self) -> None:
        token = self.var_token.get().strip()
        if not token:
            messagebox.showwarning("缺少 Token", "请先输入 TuShare Token。")
            return

        def worker():
            try:
                self.log_queue.put(("log", "开始测试 Token：调用 trade_cal。"))
                import tushare as ts
                ts.set_token(token)
                pro = ts.pro_api(token)
                df = pro.trade_cal(exchange="", start_date="20260101", end_date="20260110")
                self.log_queue.put(("log", f"Token 测试成功，返回 {len(df)} 行。"))
                self.log_queue.put(("info", "Token 有效", "Token 测试成功。"))
            except Exception:
                err = traceback.format_exc()
                self.log_queue.put(("fatal", "Token 测试失败", err))

        threading.Thread(target=worker, daemon=True).start()

    # ------------------------------------------------------------------
    # 下载控制
    # ------------------------------------------------------------------

    def start_download(self) -> None:
        if self.worker_thread and self.worker_thread.is_alive():
            messagebox.showinfo("正在运行", "当前已有下载任务正在运行。")
            return
        if not self.validate_inputs():
            return

        token = self.var_token.get().strip()
        db_path = self.var_db_path.get().strip()
        engine_path = self.var_engine_path.get().strip()
        selected_keys = self.get_selected_keys()

        if self.var_save_token_env.get():
            os.environ["TUSHARE_TOKEN"] = token
            self.append_log("已在当前进程保存 TUSHARE_TOKEN。若要永久保存，请使用 setx 或在系统环境变量中设置。")

        self.stop_event.clear()
        self.progress_var.set(0.0)
        self.status_label.configure(text="运行中")
        self.btn_start.configure(state="disabled")
        self.btn_stop.configure(state="normal")

        runner = TushareGuiRunner(
            token=token,
            db_path=db_path,
            engine_path=engine_path,
            selected_keys=selected_keys,
            log_queue=self.log_queue,
            stop_event=self.stop_event,
            only_missing=self.var_only_missing.get(),
        )
        self.worker_thread = threading.Thread(target=runner.run, daemon=True)
        self.worker_thread.start()

    def stop_download(self) -> None:
        self.stop_event.set()
        self.append_log("已请求停止。注意：当前正在运行的单个模块不会被强杀，会在模块结束后停止后续模块。")
        self.btn_stop.configure(state="disabled")

    # ------------------------------------------------------------------
    # 预留运算脚本接口
    # ------------------------------------------------------------------

    def run_analysis_script(self) -> None:
        script = self.var_analysis_script.get().strip()
        if not script:
            messagebox.showwarning("缺少脚本", "请先选择一个运算脚本。")
            return
        if not Path(script).exists():
            messagebox.showwarning("脚本不存在", f"找不到脚本：\n{script}")
            return
        if self.analysis_thread and self.analysis_thread.is_alive():
            messagebox.showinfo("正在运行", "已有运算脚本正在运行。")
            return

        db_path = self.var_db_path.get().strip()
        data_dir = str(Path(db_path).parent)

        def worker():
            try:
                self.log_queue.put(("log", "========== 运算脚本开始 =========="))
                env = os.environ.copy()
                env["TUSHARE_DB_PATH"] = db_path
                env["TUSHARE_DATA_DIR"] = data_dir
                env.setdefault("PYTHONUTF8", "1")
                env.setdefault("PYTHONIOENCODING", "utf-8")

                cmd = [sys.executable, script, "--db", db_path, "--data-dir", data_dir]
                self.log_queue.put(("log", "运行命令：" + " ".join(cmd)))

                proc = subprocess.Popen(
                    cmd,
                    cwd=data_dir,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                )
                assert proc.stdout is not None
                for line in proc.stdout:
                    if line.strip():
                        self.log_queue.put(("log", line.rstrip()))
                code = proc.wait()
                if code == 0:
                    self.log_queue.put(("done", "运算脚本运行完成"))
                else:
                    self.log_queue.put(("fatal", "运算脚本失败", f"退出码：{code}"))
                self.log_queue.put(("log", "========== 运算脚本结束 =========="))
            except Exception:
                err = traceback.format_exc()
                self.log_queue.put(("fatal", "运算脚本运行异常", err))

        self.analysis_thread = threading.Thread(target=worker, daemon=True)
        self.analysis_thread.start()

    # ------------------------------------------------------------------
    # 队列轮询
    # ------------------------------------------------------------------

    def _poll_queue(self) -> None:
        try:
            while True:
                item = self.log_queue.get_nowait()
                kind = item[0]

                if kind == "log":
                    self.append_log(item[1])

                elif kind == "progress":
                    _, current, total, label = item
                    if total <= 0:
                        pct = 0.0
                    else:
                        pct = min(100.0, max(0.0, current / total * 100.0))
                    self.progress_var.set(pct)
                    self.status_label.configure(text=f"{current}/{total}  {label}")

                elif kind == "done":
                    self.append_log(item[1])
                    self.status_label.configure(text="完成")
                    self.btn_start.configure(state="normal")
                    self.btn_stop.configure(state="disabled")
                    messagebox.showinfo("完成", item[1])

                elif kind == "info":
                    _, title, msg = item
                    self.append_log(msg)
                    messagebox.showinfo(title, msg)

                elif kind in {"error", "fatal"}:
                    _, title, err = item
                    self.append_log(f"{title}\n{err}")
                    self.status_label.configure(text="失败")
                    self.btn_start.configure(state="normal")
                    self.btn_stop.configure(state="disabled")
                    messagebox.showerror(title, err[:6000])

        except queue.Empty:
            pass
        finally:
            self.after(150, self._poll_queue)


# =============================================================================
# 5. 程序入口
# =============================================================================


def main() -> None:
    app = TushareGuiApp()
    app.mainloop()


if __name__ == "__main__":
    main()
