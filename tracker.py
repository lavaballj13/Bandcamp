#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GitHub Actions Portfolio Tracker (bands + lockup + daily/weekly email)
HARD-CODED PORTFOLIO (no portfolio.csv needed)

Email formatting (kept excluded):
- NO "Symbol" column
- NO "$Value" column
- NO "(in_lockup=True)" suffix
- NO "Auto-rebalance: ..." line
- NO "Daily email: ..." line
- NO anchor/total-value/1d-ROC header lines you previously removed

What this DOES:
- Anchors holdings to REBALANCE_DATE (default 2025-04-16) using target weights
- Uses anchor close prices to compute synthetic shares (START_CAPITAL * target / anchor_close)
- Values those shares at latest close to compute weights + breaches
- Lockup_until = anchor_used + LOCKUP_DAYS
- Sends email EVERY day (DAILY_EMAIL=1 default)
- Adds extra sensible metrics (percent-based) while keeping excluded items excluded

Deps: pandas numpy yfinance
"""

from __future__ import annotations

import io
import json
import os
import smtplib
import ssl
import sys
import time
from dataclasses import dataclass
from datetime import date, timedelta
from email.message import EmailMessage
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import yfinance as yf


# ==========================
# HARDCODED PORTFOLIO
# ==========================
# band_down/band_up are DECIMALS (0.10 = 10%)
PORT = pd.DataFrame(
    [
        {"symbol": "QQQSIM?L=3", "yf_ticker": "TQQQ", "target_w": 0.30, "band_down": 0.00, "band_up": 0.00},
        {"symbol": "DBMFSIM",    "yf_ticker": "DBMF", "target_w": 0.25, "band_down": 0.00, "band_up": 0.00},
        {"symbol": "XLESIM",     "yf_ticker": "XLE",  "target_w": 0.15, "band_down": 0.10, "band_up": 0.05},
        {"symbol": "GLDSIM",     "yf_ticker": "GLD",  "target_w": 0.15, "band_down": 0.10, "band_up": 0.05},
        {"symbol": "CASHX.1",    "yf_ticker": "SGOV", "target_w": 0.15, "band_down": 0.00, "band_up": 0.00},
    ]
).copy()

PORT["target_w"] = pd.to_numeric(PORT["target_w"], errors="coerce").fillna(0.0)
s = float(PORT["target_w"].sum())
if s <= 0:
    raise ValueError("PORT target_w sums to 0; fix configuration.")
PORT["target_w"] = PORT["target_w"] / s

for c in ["band_down", "band_up"]:
    PORT[c] = pd.to_numeric(PORT[c], errors="coerce").fillna(0.0).clip(lower=0.0, upper=0.95)

PORT = PORT.reset_index(drop=True)


# ==========================
# ENV helpers
# ==========================
def _env_float(name: str, default: str) -> float:
    v = os.getenv(name)
    if v is None or str(v).strip() == "":
        return float(default)
    return float(v)


def _env_int(name: str, default: str) -> int:
    v = os.getenv(name)
    if v is None or str(v).strip() == "":
        return int(default)
    return int(v)


def _env_bool(name: str, default: str = "0") -> bool:
    v = os.getenv(name, default)
    return str(v).strip() == "1"


def _env_date(name: str, default_iso: str) -> date:
    v = os.getenv(name)
    if v is None or str(v).strip() == "":
        v = default_iso
    return pd.to_datetime(v, errors="raise").date()


# ==========================
# CONFIG
# ==========================
START_CAPITAL = _env_float("START_CAPITAL", "100000.0")
LOCKUP_DAYS = _env_int("LOCKUP_DAYS", os.getenv("TAX_LOCKUP_DAYS", "365"))
ENFORCE_LOCKUP = _env_bool("ENFORCE_LOCKUP", "1")

MONTHLY_CONTRIBUTION = _env_float("MONTHLY_CONTRIBUTION", "2000.0")  # used for attachment
DBG = _env_bool("TRACKER_DEBUG", "0")

# Anchoring date (requested)
REBALANCE_DATE = _env_date("REBALANCE_DATE", "2025-04-16")

# Always send daily email
DAILY_EMAIL = _env_bool("DAILY_EMAIL", "1")

EPS = 1e-12


# ==========================
# FILES
# ==========================
HOLDINGS_JSON = "holdings_state.json"
REBAL_JSON = "rebalance_state.json"
HISTORY_CSV = "history.csv"
STATUS_JSON = "status.json"
SIGNALS_JSON = "signals_state.json"


# ==========================
# EMAIL ENV (GitHub Secrets)
# ==========================
SMTP_HOST = os.getenv("SMTP_HOST")
SMTP_PORT = _env_int("SMTP_PORT", "587")
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASS = os.getenv("SMTP_PASS")
MAIL_FROM = os.getenv("MAIL_FROM")
MAIL_TO = os.getenv("MAIL_TO")
EMAIL_DISABLE = _env_bool("EMAIL_DISABLE", "0")


# ==========================
# LOGGING
# ==========================
def _dbg(*a):
    if DBG:
        print("[dbg]", *a, flush=True)


def _warn(*a):
    print("[warn]", *a, flush=True)


def _info(*a):
    print("[info]", *a, flush=True)


def _err(*a):
    print("[error]", *a, flush=True)


# ==========================
# JSON helpers
# ==========================
def load_json(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        _warn(f"{path} unreadable; treating as empty. err={repr(e)}")
        return {}


def save_json(path: str, obj: dict) -> None:
    Path(path).write_text(json.dumps(obj, indent=2), encoding="utf-8")


def _parse_date(s: Optional[str]) -> Optional[date]:
    if not s:
        return None
    try:
        return pd.to_datetime(s).date()
    except Exception:
        return None


# ==========================
# EMAIL
# ==========================
def email_send(subject: str, body: str, attachments: Optional[List[Tuple[str, pd.DataFrame]]] = None) -> None:
    if EMAIL_DISABLE:
        _warn("EMAIL_DISABLE=1 → skipping email send")
        return

    if not all([SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS, MAIL_FROM, MAIL_TO]):
        _warn("Email not configured (missing SMTP_* or MAIL_*).")
        _warn(f"SMTP_HOST set? {bool(SMTP_HOST)}  SMTP_USER set? {bool(SMTP_USER)}  MAIL_TO set? {bool(MAIL_TO)}")
        return

    try:
        msg = EmailMessage()
        msg["From"] = MAIL_FROM
        msg["To"] = MAIL_TO
        msg["Subject"] = subject
        msg.set_content(body)

        for name, df in (attachments or []):
            buf = io.StringIO()
            if isinstance(df, pd.DataFrame) and len(df) > 0:
                df.to_csv(buf, index=False)
            else:
                buf.write("")
            msg.add_attachment(
                buf.getvalue().encode("utf-8"),
                maintype="text",
                subtype="csv",
                filename=name,
            )

        ctx = ssl.create_default_context()
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=30) as s:
            s.starttls(context=ctx)
            s.login(SMTP_USER, SMTP_PASS)
            s.send_message(msg)

        _info("email sent:", subject)
    except Exception as e:
        _warn("email_send failed:", repr(e))


# ==========================
# yfinance
# ==========================
def _yf_download(tickers: List[str], period: str = "5y", interval: str = "1d", tries: int = 3) -> pd.DataFrame:
    last_err = None
    for k in range(tries):
        try:
            df = yf.download(
                tickers=tickers,
                period=period,
                interval=interval,
                auto_adjust=False,
                progress=False,
                group_by="column",
                threads=True,
            )
            if isinstance(df, pd.DataFrame) and len(df) >= 1:
                return df
        except Exception as e:
            last_err = e
        time.sleep(1.0 + 0.75 * k)
    _warn(f"yfinance download empty after {tries} tries; err={last_err}")
    return pd.DataFrame()


def _get_field_frame(frame: pd.DataFrame, field: str) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or len(frame) == 0:
        raise KeyError("empty frame")

    if isinstance(frame.columns, pd.MultiIndex):
        fields = set(frame.columns.get_level_values(0))
        if field in fields:
            return frame[field]
        if field == "Adj Close" and "Close" in fields:
            _warn("Adj Close missing; using Close as fallback")
            return frame["Close"]
        raise KeyError(f"{field} not found in fields: {sorted(fields)}")

    if field in frame.columns:
        return frame[[field]]
    if field == "Adj Close" and "Close" in frame.columns:
        _warn("Adj Close missing; using Close as fallback")
        return frame[["Close"]]
    raise KeyError(f"{field} not found in columns: {frame.columns.tolist()}")


def _last_non_nan_prices(frame: pd.DataFrame, field: str, tickers: List[str]) -> Tuple[pd.Timestamp, pd.Series]:
    if not isinstance(frame, pd.DataFrame) or len(frame) == 0 or not isinstance(frame.index, pd.DatetimeIndex):
        raise RuntimeError("Invalid yfinance frame (empty or missing DatetimeIndex).")

    as_of_dt = pd.Timestamp(frame.index[-1]).tz_localize(None)
    sub = _get_field_frame(frame, field)

    if isinstance(frame.columns, pd.MultiIndex):
        prices: Dict[str, float] = {}
        for t in tickers:
            if t in sub.columns:
                vv = sub[t].dropna()
                prices[t] = float(vv.iloc[-1]) if len(vv) > 0 else np.nan
            else:
                prices[t] = np.nan
        return as_of_dt, pd.Series(prices, dtype="float64").reindex(tickers)

    vv = sub.iloc[:, 0].dropna()
    v = float(vv.iloc[-1]) if len(vv) > 0 else np.nan
    return as_of_dt, pd.Series({tickers[0]: v}, dtype="float64").reindex(tickers)


def _anchor_prices_all_tickers(close_df: pd.DataFrame, anchor_req: date, tickers: List[str]) -> Tuple[date, pd.Series]:
    """
    Choose an anchor date <= anchor_req where ALL tickers have non-NaN close.
    Returns (anchor_used_date, prices_series indexed by tickers).
    """
    if not isinstance(close_df.index, pd.DatetimeIndex) or len(close_df) == 0:
        raise RuntimeError("No close data to compute anchor prices.")

    df = close_df.copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df.sort_index().reindex(columns=tickers)

    df = df[df.index.date <= anchor_req]
    df = df.dropna(how="any")
    if len(df) == 0:
        raise RuntimeError(f"No common anchor close available on/before {anchor_req.isoformat()} for all tickers.")

    row = df.iloc[-1].astype(float)
    anchor_used = df.index[-1].date()
    return anchor_used, row


# ==========================
# REBALANCE STATE
# ==========================
def load_rebalance_state() -> dict:
    return load_json(REBAL_JSON)


def set_rebalance_state(*, last_rebalance_date: date, lockup_until: date, reason: str) -> None:
    st = {
        "last_rebalance_date": last_rebalance_date.isoformat(),
        "lockup_until": lockup_until.isoformat(),
        "lockup_days": int(LOCKUP_DAYS),
        "lockup_enforced": bool(ENFORCE_LOCKUP),
        "reason": str(reason),
    }
    save_json(REBAL_JSON, st)


def lockup_status(as_of: date) -> Tuple[bool, Optional[date]]:
    st = load_rebalance_state()
    until = _parse_date(st.get("lockup_until"))
    if not ENFORCE_LOCKUP or until is None:
        return (False, until)
    return (as_of < until, until)


# ==========================
# HOLDINGS STATE (anchored)
# ==========================
def load_holdings() -> dict:
    return load_json(HOLDINGS_JSON)


def save_holdings(holdings: dict) -> None:
    save_json(HOLDINGS_JSON, holdings)


def ensure_holdings_anchored(port: pd.DataFrame, anchor_used: date, anchor_close_by_yf: pd.Series) -> dict:
    """
    Ensure holdings_state.json represents anchor-date synthetic shares computed
    from START_CAPITAL and target weights. If existing anchor differs, overwrite.
    """
    h = load_holdings()
    existing_anchor = _parse_date(h.get("anchor_date") or h.get("as_of"))

    tickers = port["yf_ticker"].tolist()
    px = anchor_close_by_yf.reindex(tickers).astype(float)

    if px.isna().any() or (px <= 0).any():
        bad = px[px.isna() | (px <= 0)]
        raise RuntimeError(f"Cannot anchor holdings; bad anchor prices: {bad.to_dict()}")

    if isinstance(h.get("shares"), dict) and len(h["shares"]) > 0 and existing_anchor == anchor_used:
        return h

    tgt = port["target_w"].to_numpy(float)
    shares: Dict[str, float] = {}
    for yf_t, w, p in zip(tickers, tgt, px.to_numpy(float)):
        shares[yf_t] = float((START_CAPITAL * float(w)) / float(p))

    h_new = {
        "anchor_date": anchor_used.isoformat(),
        "capital_init": float(START_CAPITAL),
        "shares": shares,
    }
    save_holdings(h_new)
    _info(f"Anchored holdings_state.json to {anchor_used.isoformat()} using START_CAPITAL={START_CAPITAL:,.2f}")

    lock_until = anchor_used + timedelta(days=int(LOCKUP_DAYS)) if ENFORCE_LOCKUP else anchor_used
    set_rebalance_state(last_rebalance_date=anchor_used, lockup_until=lock_until, reason="anchor_init")
    return h_new


# ==========================
# ALLOCATION
# ==========================
@dataclass
class AllocationSnapshot:
    as_of: date
    total_value: float
    table: pd.DataFrame


def compute_allocation(port: pd.DataFrame, shares_by_yf: Dict[str, float], close_by_yf: pd.Series, as_of: date) -> AllocationSnapshot:
    df = port.copy()
    df["shares"] = df["yf_ticker"].map(lambda t: float(shares_by_yf.get(t, 0.0)))
    df["price"] = df["yf_ticker"].map(lambda t: float(close_by_yf.get(t, np.nan)))

    df["value"] = df["shares"] * df["price"]
    df["value"] = df["value"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    total = float(df["value"].sum())
    df["weight_now"] = (df["value"] / total) if total > 0 else 0.0

    df["lo"] = (df["target_w"] - df["band_down"]).clip(lower=0.0)
    df["hi"] = (df["target_w"] + df["band_up"]).clip(upper=1.0)
    df["delta_w"] = df["weight_now"] - df["target_w"]

    status = []
    for _, r in df.iterrows():
        bd = float(r["band_down"])
        bu = float(r["band_up"])
        wn = float(r["weight_now"])
        if bd <= 0.0 and bu <= 0.0:
            status.append("NO_BAND")
        elif wn < float(r["lo"]) - 1e-9:
            status.append("LOW")
        elif wn > float(r["hi"]) + 1e-9:
            status.append("HIGH")
        else:
            status.append("OK")
    df["status"] = status

    df = df.sort_values("value", ascending=False).reset_index(drop=True)
    return AllocationSnapshot(as_of=as_of, total_value=total, table=df)


def rebalance_deltas(tbl: pd.DataFrame) -> pd.DataFrame:
    total = float(tbl["value"].sum())
    df = tbl.copy()
    df["target_$"] = df["target_w"] * total
    df["$delta_to_target"] = df["target_$"] - df["value"]
    return df[["yf_ticker", "weight_now", "target_w", "$delta_to_target"]].sort_values(
        "$delta_to_target", ascending=False
    )


def suggest_contribution_split(tbl: pd.DataFrame, contribution: float) -> pd.DataFrame:
    df = tbl.copy()
    df["deficit"] = np.maximum(0.0, df["target_w"] - df["weight_now"])

    if contribution <= 0:
        df["buy_$"] = 0.0
        return df[["yf_ticker", "weight_now", "target_w", "deficit", "buy_$"]]

    d = df["deficit"].to_numpy(float)
    if float(d.sum()) <= 1e-12:
        w = df["target_w"].to_numpy(float)
        w = w / max(EPS, float(w.sum()))
    else:
        w = d / float(d.sum())

    df["buy_$"] = contribution * w
    return df[["yf_ticker", "weight_now", "target_w", "deficit", "buy_$"]].sort_values("buy_$", ascending=False)


# ==========================
# SIGNALS STATE
# ==========================
def load_signals_state() -> dict:
    return load_json(SIGNALS_JSON)


def save_signals_state(state: dict) -> None:
    save_json(SIGNALS_JSON, state)


def current_signal_map(tbl: pd.DataFrame) -> dict:
    out = {}
    for _, r in tbl.iterrows():
        if r["status"] in ("LOW", "HIGH"):
            out[str(r["yf_ticker"])] = str(r["status"])
    return out


def update_signal_timestamps(prev_state: dict, now_signals: dict, as_of: date) -> dict:
    first_seen = dict(prev_state.get("first_seen", {}) or {})
    last_seen = dict(prev_state.get("last_seen", {}) or {})

    for yf_t in now_signals.keys():
        if yf_t not in first_seen:
            first_seen[yf_t] = as_of.isoformat()
        last_seen[yf_t] = as_of.isoformat()

    return {
        "as_of": as_of.isoformat(),
        "signals": now_signals,
        "first_seen": first_seen,
        "last_seen": last_seen,
    }


# ==========================
# HISTORY (schema hardened)
# ==========================
def _normalize_history_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or not isinstance(df, pd.DataFrame) or len(df.columns) == 0:
        return pd.DataFrame(columns=["date", "total_value"])

    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]

    if "date" not in out.columns:
        for cand in ("Date", "DATE", "day", "Day", "timestamp", "Timestamp"):
            if cand in out.columns:
                out = out.rename(columns={cand: "date"})
                break

    if "total_value" not in out.columns:
        for cand in ("total", "Total", "portfolio_value", "PortfolioValue", "value", "Value", "equity", "Equity"):
            if cand in out.columns:
                out = out.rename(columns={cand: "total_value"})
                break

    if "total_value" not in out.columns:
        numeric_cols = []
        for c in out.columns:
            if c == "date":
                continue
            s = pd.to_numeric(out[c], errors="coerce")
            if s.notna().any():
                numeric_cols.append(c)
        if len(numeric_cols) == 1:
            out = out.rename(columns={numeric_cols[0]: "total_value"})

    if "date" not in out.columns:
        return pd.DataFrame(columns=["date", "total_value"])

    if "total_value" not in out.columns:
        _warn("history.csv missing total_value column; columns were:", out.columns.tolist())
        out["total_value"] = np.nan

    return out[["date", "total_value"]].copy()


def _collapse_history_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=["date", "total_value"])

    df = _normalize_history_columns(df)

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["total_value"] = pd.to_numeric(df["total_value"], errors="coerce")

    df = df.dropna(subset=["date"]).sort_values("date")
    if len(df) == 0:
        return pd.DataFrame(columns=["date", "total_value"])

    g = df.groupby(df["date"].dt.date, as_index=False).last()
    g["date"] = pd.to_datetime(g["date"], errors="coerce")
    return g


def update_history(as_of: date, total_value: float) -> float:
    cols = ["date", "total_value"]
    day = as_of.isoformat()
    new_row = pd.DataFrame([{"date": day, "total_value": float(total_value)}], columns=cols)

    path = Path(HISTORY_CSV)
    prev_val = np.nan

    if path.exists():
        try:
            raw = pd.read_csv(path)
        except Exception as e:
            _warn("history.csv unreadable; resetting. err=", repr(e))
            raw = pd.DataFrame(columns=cols)

        df = _collapse_history_df(raw)
        df_wo_today = df[df["date"].dt.date.astype(str) != day].sort_values("date")
        if len(df_wo_today) > 0 and "total_value" in df_wo_today.columns:
            try:
                prev_val = float(df_wo_today["total_value"].iloc[-1])
            except Exception:
                prev_val = np.nan

        df = pd.concat([df_wo_today, new_row], ignore_index=True)
    else:
        df = new_row

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["total_value"] = pd.to_numeric(df["total_value"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    df[cols].to_csv(path, index=False)

    roc_1d = 0.0
    if np.isfinite(prev_val) and prev_val > 0:
        roc_1d = float(total_value / prev_val - 1.0)
    return float(roc_1d)


# ==========================
# EMAIL TABLE FORMATTING (aligned, no Symbol, no $Value)
# ==========================
def _pct_str(x: float) -> str:
    return f"{x*100:6.2f}%"


def format_alloc_table(tbl: pd.DataFrame) -> str:
    """
    Columns: YF | Now | Target | Band | Status
    """
    df = tbl.copy()

    band_strs: List[str] = []
    for _, r in df.iterrows():
        bd = float(r["band_down"])
        bu = float(r["band_up"])
        if bd <= 0 and bu <= 0:
            band_strs.append("—")
        else:
            band_strs.append(f"[{r['lo']*100:5.2f}%..{r['hi']*100:5.2f}%]")
    df["band_str"] = band_strs

    yf_w = max(2, int(df["yf_ticker"].astype(str).map(len).max() if len(df) else 2))
    band_w = max(len("Band"), int(df["band_str"].map(len).max() if len(df) else len("Band")))
    st_w = max(len("Status"), int(df["status"].astype(str).map(len).max() if len(df) else len("Status")))

    header = f"{'YF':<{yf_w}}  {'Now':>9}  {'Target':>9}  {'Band':<{band_w}}  {'Status':<{st_w}}"
    lines = [header]

    for _, r in df.iterrows():
        lines.append(
            f"{str(r['yf_ticker']):<{yf_w}}  "
            f"{_pct_str(float(r['weight_now'])):>9}  "
            f"{_pct_str(float(r['target_w'])):>9}  "
            f"{str(r['band_str']):<{band_w}}  "
            f"{str(r['status']):<{st_w}}"
        )
    return "\n".join(lines)


# ==========================
# EXTRA METRICS (since rebalance)
# ==========================
@dataclass
class Metrics:
    ret_since: Optional[float]
    cagr: Optional[float]
    vol_ann: Optional[float]
    sharpe0: Optional[float]
    max_dd: Optional[float]
    best_day: Optional[Tuple[str, float]]
    worst_day: Optional[Tuple[str, float]]
    ret_5d: Optional[float]
    ret_21d: Optional[float]
    ret_mtd: Optional[float]
    ret_ytd: Optional[float]
    days_since: int
    biggest_over: Optional[Tuple[str, float]]
    biggest_under: Optional[Tuple[str, float]]


def _compute_metrics(
    *,
    anchor_value: float,
    total_value: float,
    series_values: pd.Series,
    alloc_tbl: pd.DataFrame,
    anchor_used: date,
    as_of: date,
) -> Metrics:
    ret_since = (total_value / anchor_value - 1.0) if anchor_value > 0 else None

    days_since = int((as_of - anchor_used).days)

    daily_rets = series_values.pct_change().dropna()

    cagr = None
    if anchor_value > 0 and days_since > 0:
        yrs = days_since / 365.25
        if yrs > 0:
            cagr = (total_value / anchor_value) ** (1.0 / yrs) - 1.0

    vol_ann = None
    sharpe0 = None
    if len(daily_rets) >= 2:
        vol_d = float(daily_rets.std())
        vol_ann = vol_d * float(np.sqrt(252.0))
        mu_ann = float(daily_rets.mean()) * 252.0
        if vol_ann and vol_ann > 0:
            sharpe0 = mu_ann / vol_ann

    max_dd = None
    if len(series_values) >= 2:
        peak = series_values.cummax()
        dd = series_values / peak - 1.0
        max_dd = float(dd.min())

    best_day = None
    worst_day = None
    if len(daily_rets) >= 1:
        i_best = daily_rets.idxmax()
        i_worst = daily_rets.idxmin()
        best_day = (pd.Timestamp(i_best).date().isoformat(), float(daily_rets.loc[i_best]))
        worst_day = (pd.Timestamp(i_worst).date().isoformat(), float(daily_rets.loc[i_worst]))

    ret_5d = None
    ret_21d = None
    if len(series_values) >= 6:
        ret_5d = float(series_values.iloc[-1] / series_values.iloc[-6] - 1.0)
    if len(series_values) >= 22:
        ret_21d = float(series_values.iloc[-1] / series_values.iloc[-22] - 1.0)

    ret_mtd = None
    ret_ytd = None
    if len(series_values) >= 2:
        s = series_values.copy()
        s.index = pd.to_datetime(s.index)

        last_ts = s.index[-1]
        m_start = last_ts.to_period("M")
        y_start = last_ts.year

        m = s[s.index.to_period("M") == m_start]
        if len(m) >= 2:
            ret_mtd = float(m.iloc[-1] / m.iloc[0] - 1.0)

        y = s[s.index.year == y_start]
        if len(y) >= 2:
            ret_ytd = float(y.iloc[-1] / y.iloc[0] - 1.0)

    biggest_over = None
    biggest_under = None
    if "delta_w" in alloc_tbl.columns and len(alloc_tbl) > 0:
        over = alloc_tbl.sort_values("delta_w", ascending=False).iloc[0]
        under = alloc_tbl.sort_values("delta_w", ascending=True).iloc[0]
        biggest_over = (str(over["yf_ticker"]), float(over["delta_w"]))
        biggest_under = (str(under["yf_ticker"]), float(under["delta_w"]))

    return Metrics(
        ret_since=ret_since,
        cagr=cagr,
        vol_ann=vol_ann,
        sharpe0=sharpe0,
        max_dd=max_dd,
        best_day=best_day,
        worst_day=worst_day,
        ret_5d=ret_5d,
        ret_21d=ret_21d,
        ret_mtd=ret_mtd,
        ret_ytd=ret_ytd,
        days_since=days_since,
        biggest_over=biggest_over,
        biggest_under=biggest_under,
    )


# ==========================
# DAILY JOB
# ==========================
def daily_job(*, email_subject_prefix: str = "Portfolio — Daily update") -> None:
    tickers = PORT["yf_ticker"].tolist()

    data = _yf_download(tickers, period="5y", interval="1d", tries=3)
    if not isinstance(data, pd.DataFrame) or len(data) == 0 or not isinstance(data.index, pd.DatetimeIndex):
        raise RuntimeError("yfinance returned no usable data; cannot run daily job.")

    close_df = _get_field_frame(data, "Close")
    close_df.index = pd.to_datetime(close_df.index).tz_localize(None)
    close_df = close_df.sort_index().reindex(columns=tickers)

    anchor_used, anchor_px = _anchor_prices_all_tickers(close_df, REBALANCE_DATE, tickers)
    as_of_dt, latest_px = _last_non_nan_prices(data, "Close", tickers)
    as_of = as_of_dt.date()

    if latest_px.isna().any():
        bad = latest_px[latest_px.isna()]
        raise RuntimeError(f"Missing latest Close prices for: {bad.index.tolist()}")

    holdings = ensure_holdings_anchored(PORT, anchor_used, anchor_px)
    shares_by_yf = holdings.get("shares", {}) or {}

    snap = compute_allocation(PORT, shares_by_yf, latest_px, as_of)

    anchor_value = float(START_CAPITAL)
    total_value = float(snap.total_value)

    # Portfolio value series since anchor (for metrics)
    px_slice = close_df[close_df.index.date >= anchor_used].copy()
    px_slice = px_slice.ffill().dropna(how="any")
    if len(px_slice) == 0:
        raise RuntimeError("No price history slice available since anchor to compute metrics.")

    w_shares = pd.Series({t: float(shares_by_yf.get(t, 0.0)) for t in tickers}, dtype="float64").reindex(tickers).fillna(0.0)
    series_values = (px_slice * w_shares).sum(axis=1)
    if not isinstance(series_values, pd.Series) or len(series_values) < 2:
        raise RuntimeError("Portfolio value series too short to compute metrics.")

    metrics = _compute_metrics(
        anchor_value=anchor_value,
        total_value=total_value,
        series_values=series_values,
        alloc_tbl=snap.table,
        anchor_used=anchor_used,
        as_of=as_of,
    )

    # Keep writing history.csv (still useful for artifacts; not shown in email)
    update_history(as_of, total_value)

    # Signals
    sig_now = current_signal_map(snap.table)
    prev_sig_state = load_signals_state()
    prev_sig_map = dict(prev_sig_state.get("signals", {}) or {})
    changed = (sig_now != prev_sig_map)
    new_sig_state = update_signal_timestamps(prev_sig_state, sig_now, as_of)
    save_signals_state(new_sig_state)

    # Lockup status (anchored)
    in_lock, lock_until = lockup_status(as_of)

    # status.json (debug + audit)
    status = {
        "as_of": as_of.isoformat(),
        "as_of_timestamp": str(as_of_dt),
        "rebalance_date_requested": REBALANCE_DATE.isoformat(),
        "anchor_used": anchor_used.isoformat(),
        "anchor_value": float(anchor_value),
        "total_value": float(total_value),
        "return_since_rebalance": float(metrics.ret_since) if metrics.ret_since is not None else None,
        "cagr_since_rebalance": float(metrics.cagr) if metrics.cagr is not None else None,
        "vol_ann": float(metrics.vol_ann) if metrics.vol_ann is not None else None,
        "sharpe0": float(metrics.sharpe0) if metrics.sharpe0 is not None else None,
        "max_drawdown": float(metrics.max_dd) if metrics.max_dd is not None else None,
        "lockup_days": int(LOCKUP_DAYS),
        "lockup_enforced": bool(ENFORCE_LOCKUP),
        "lockup_until": (lock_until.isoformat() if lock_until else None),
        "in_lockup": bool(in_lock),
        "daily_email": bool(DAILY_EMAIL),
        "signals": sig_now,
        "signals_first_seen": new_sig_state.get("first_seen", {}),
        "signals_last_seen": new_sig_state.get("last_seen", {}),
        "portfolio": PORT.to_dict(orient="records"),
        "allocation": snap.table.to_dict(orient="records"),
        "anchor_prices": anchor_px.to_dict(),
        "latest_prices": latest_px.to_dict(),
    }
    Path(STATUS_JSON).write_text(json.dumps(status, indent=2), encoding="utf-8")

    # ==========================
    # EMAIL BODY (kept exclusions)
    # ==========================
    header = [f"As of: {as_of.isoformat()} (EOD Close)"]

    # Return since rebalance (kept)
    if metrics.ret_since is not None and np.isfinite(metrics.ret_since):
        header.append(f"Return since rebalance: {metrics.ret_since*100:+.2f}%")
    else:
        header.append("Return since rebalance: n/a")

    # Extra metrics (percent-based; no $ totals / no anchor lines)
    header.append(f"Days since rebalance: {metrics.days_since}")

    if metrics.cagr is not None and np.isfinite(metrics.cagr):
        header.append(f"CAGR since rebalance: {metrics.cagr*100:+.2f}%")
    else:
        header.append("CAGR since rebalance: n/a")

    if metrics.max_dd is not None and np.isfinite(metrics.max_dd):
        header.append(f"Max drawdown since rebalance: {metrics.max_dd*100:+.2f}%")
    else:
        header.append("Max drawdown since rebalance: n/a")

    if metrics.vol_ann is not None and np.isfinite(metrics.vol_ann):
        header.append(f"Volatility (ann.): {metrics.vol_ann*100:.2f}%")
    else:
        header.append("Volatility (ann.): n/a")

    if metrics.sharpe0 is not None and np.isfinite(metrics.sharpe0):
        header.append(f"Sharpe (0% rf): {metrics.sharpe0:.2f}")
    else:
        header.append("Sharpe (0% rf): n/a")

    # short horizon returns
    if metrics.ret_5d is not None and np.isfinite(metrics.ret_5d):
        header.append(f"~5d return: {metrics.ret_5d*100:+.2f}%")
    else:
        header.append("~5d return: n/a")

    if metrics.ret_21d is not None and np.isfinite(metrics.ret_21d):
        header.append(f"~21d return: {metrics.ret_21d*100:+.2f}%")
    else:
        header.append("~21d return: n/a")

    if metrics.ret_mtd is not None and np.isfinite(metrics.ret_mtd):
        header.append(f"MTD: {metrics.ret_mtd*100:+.2f}%")
    else:
        header.append("MTD: n/a")

    if metrics.ret_ytd is not None and np.isfinite(metrics.ret_ytd):
        header.append(f"YTD: {metrics.ret_ytd*100:+.2f}%")
    else:
        header.append("YTD: n/a")

    # best/worst day
    if metrics.best_day is not None:
        header.append(f"Best day: {metrics.best_day[0]}  {metrics.best_day[1]*100:+.2f}%")
    else:
        header.append("Best day: n/a")

    if metrics.worst_day is not None:
        header.append(f"Worst day: {metrics.worst_day[0]}  {metrics.worst_day[1]*100:+.2f}%")
    else:
        header.append("Worst day: n/a")

    # biggest over/under
    if metrics.biggest_over is not None:
        header.append(f"Biggest overweight: {metrics.biggest_over[0]}  {metrics.biggest_over[1]*100:+.2f}% vs target")
    else:
        header.append("Biggest overweight: n/a")

    if metrics.biggest_under is not None:
        header.append(f"Biggest underweight: {metrics.biggest_under[0]}  {metrics.biggest_under[1]*100:+.2f}% vs target")
    else:
        header.append("Biggest underweight: n/a")

    # Lockup lines (no "(in_lockup=True)")
    header.append(f"Lockup: {'ON' if ENFORCE_LOCKUP else 'OFF'} ({LOCKUP_DAYS} calendar days)")
    header.append(f"Locked until: {lock_until.isoformat() if lock_until else '—'}")

    body = "\n".join(header) + "\n\n"
    body += "Allocation / Bands:\n"
    body += format_alloc_table(snap.table) + "\n\n"

    body += "Price change since rebalance (anchor close → latest close):\n"
    for yf_t in tickers:
        a = float(anchor_px.get(yf_t, np.nan))
        b = float(latest_px.get(yf_t, np.nan))
        if np.isfinite(a) and a > 0 and np.isfinite(b):
            pct = (b / a - 1.0) * 100.0
            body += f"- {yf_t}: {pct:+.2f}%  ({a:.2f} → {b:.2f})\n"
        else:
            body += f"- {yf_t}: n/a\n"

    if sig_now:
        body += "\n⚠️ Band Breaches:\n"
        for yf_t, st in sig_now.items():
            r = snap.table[snap.table["yf_ticker"] == yf_t].iloc[0]
            first_seen = new_sig_state.get("first_seen", {}).get(yf_t, "—")
            last_seen = new_sig_state.get("last_seen", {}).get(yf_t, "—")
            bd = float(r["band_down"])
            bu = float(r["band_up"])
            band_txt = "—" if (bd <= 0 and bu <= 0) else f"[{r['lo']*100:.2f}%..{r['hi']*100:.2f}%]"
            body += (
                f"- {yf_t}: {st}  now={r['weight_now']*100:.2f}%  band={band_txt}  "
                f"target={r['target_w']*100:.2f}%  (first_seen={first_seen}, last_seen={last_seen})\n"
            )

    # Attachments
    reb_df = rebalance_deltas(snap.table)
    contrib_df = suggest_contribution_split(snap.table, MONTHLY_CONTRIBUTION)

    breaches_exist = bool(sig_now)
    should_email = bool(DAILY_EMAIL)

    if should_email:
        subj = email_subject_prefix
        if breaches_exist:
            subj = f"{email_subject_prefix} (signals)"
        email_send(
            subject=subj,
            body=body,
            attachments=[
                ("allocation.csv", snap.table),
                ("rebalance_to_target_deltas.csv", reb_df),
                (f"contribution_split_${int(MONTHLY_CONTRIBUTION)}.csv", contrib_df),
            ],
        )
    else:
        _info("No email sent (DAILY_EMAIL=0).")


# ==========================
# WEEKLY JOB
# ==========================
def weekly_job() -> None:
    # Same content, different subject line.
    daily_job(email_subject_prefix="Portfolio — Weekly recap")


# ==========================
# MAIN
# ==========================
def main() -> None:
    mode = (sys.argv[1] if len(sys.argv) > 1 else "daily").strip().lower()
    if mode not in ("daily", "weekly"):
        print("Usage: python tracker.py [daily|weekly]")
        sys.exit(2)

    if DBG:
        _dbg(
            "ENV summary:",
            f"EMAIL_DISABLE={EMAIL_DISABLE}",
            f"DAILY_EMAIL={DAILY_EMAIL}",
            f"ENFORCE_LOCKUP={ENFORCE_LOCKUP}",
            f"LOCKUP_DAYS={LOCKUP_DAYS}",
            f"REBALANCE_DATE={REBALANCE_DATE.isoformat()}",
        )

    try:
        if mode == "daily":
            daily_job()
        else:
            weekly_job()
    except Exception as e:
        _err("tracker crashed:", repr(e))
        raise


if __name__ == "__main__":
    main()
