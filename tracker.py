#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GitHub Actions Portfolio Tracker (bands + lockup + optional auto-rebalance + email)
HARD-CODED PORTFOLIO (no portfolio.csv needed)

Key behavior (your request):
- Uses "last rebalance date" = 2025-04-16 by default (overrideable via env)
- Computes synthetic shares using PRICES ON (or last trading day BEFORE) the rebalance date
- Tracks portfolio from that rebalance date up to the latest available close (today run)
- Lockup is computed from the rebalance date (rebalance_date + LOCKUP_DAYS)
- Sends DAILY email (not only on breaches) when DAILY_EMAIL=1 (default 1)
- Bands are applied exactly as configured below

Artifacts:
- holdings_state.json     (shares by yf_ticker + anchor_date)
- rebalance_state.json    (last_rebalance_date + lockup_until + reason)
- signals_state.json      (current signals + first_seen/last_seen)
- status.json             (latest snapshot)
- history.csv             (daily total_value + 1d ROC, idempotent per date)
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

# Normalize targets defensively (should already sum to 1)
PORT["target_w"] = pd.to_numeric(PORT["target_w"], errors="coerce").fillna(0.0)
s = float(PORT["target_w"].sum())
if s <= 0:
    raise ValueError("PORT target_w sums to 0; fix configuration.")
PORT["target_w"] = PORT["target_w"] / s

for c in ["band_down", "band_up"]:
    PORT[c] = pd.to_numeric(PORT[c], errors="coerce").fillna(0.0).clip(lower=0.0, upper=0.95)

PORT = PORT.reset_index(drop=True)


# ==========================
# ENV HELPERS
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


# ==========================
# ENV CONFIG
# ==========================
START_CAPITAL = _env_float("START_CAPITAL", "100000.0")

LOCKUP_DAYS = _env_int("LOCKUP_DAYS", os.getenv("TAX_LOCKUP_DAYS", "365"))
ENFORCE_LOCKUP = _env_bool("ENFORCE_LOCKUP", "1")

AUTO_REBALANCE = _env_bool("AUTO_REBALANCE", "0")
MONTHLY_CONTRIBUTION = _env_float("MONTHLY_CONTRIBUTION", "2000.0")

# Email behavior:
# - DAILY_EMAIL=1 -> always send daily update email
# - DAILY_EMAIL=0 -> only send when (breach changed OR did_rebalance OR ALWAYS_ALERT_ON_BREACH and breach exists)
DAILY_EMAIL = _env_bool("DAILY_EMAIL", "1")
ALWAYS_ALERT_ON_BREACH = _env_bool("ALWAYS_ALERT_ON_BREACH", "0")

DBG = _env_bool("TRACKER_DEBUG", "0")
EPS = 1e-12

# Rebalance anchor controls:
LAST_REBALANCE_DATE_STR = os.getenv("LAST_REBALANCE_DATE", "2025-04-16")
FORCE_LAST_REBALANCE_DATE = _env_bool("FORCE_LAST_REBALANCE_DATE", "1")

# If 1 (default), holdings will be (re)built to match targets at last_rebalance_date prices.
# This ensures performance from the rebalance date to today uses the correct anchor.
FORCE_HOLDINGS_TO_REBALANCE_DATE = _env_bool("FORCE_HOLDINGS_TO_REBALANCE_DATE", "1")


# ==========================
# FILES
# ==========================
HOLDINGS_JSON = "holdings_state.json"
REBAL_JSON = "rebalance_state.json"
HISTORY_CSV = "history.csv"
STATUS_JSON = "status.json"
SIGNALS_JSON = "signals_state.json"


# ==========================
# EMAIL (GitHub Secrets)
# ==========================
SMTP_HOST = os.getenv("SMTP_HOST")
SMTP_PORT = _env_int("SMTP_PORT", "587")  # robust (handles empty string)
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
# yfinance helpers
# ==========================
def _yf_download(
    tickers: List[str],
    *,
    start: Optional[date] = None,
    end: Optional[date] = None,
    period: str = "60d",
    interval: str = "1d",
    tries: int = 3,
) -> pd.DataFrame:
    """
    If start/end are provided, uses date-range download; else uses period.
    """
    last_err = None
    for k in range(tries):
        try:
            kwargs = dict(
                tickers=tickers,
                interval=interval,
                auto_adjust=False,
                progress=False,
                group_by="column",
                threads=True,
            )
            if start is not None:
                kwargs["start"] = pd.Timestamp(start)
            if end is not None:
                kwargs["end"] = pd.Timestamp(end)
            if start is None and end is None:
                kwargs["period"] = period

            df = yf.download(**kwargs)
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
    """
    Latest available close in frame, per ticker.
    """
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


def _prices_on_or_before(frame: pd.DataFrame, field: str, tickers: List[str], on_or_before: date) -> Tuple[pd.Timestamp, pd.Series]:
    """
    Prices on the given calendar date, or (if market closed) last trading day BEFORE it.
    """
    if not isinstance(frame, pd.DataFrame) or len(frame) == 0 or not isinstance(frame.index, pd.DatetimeIndex):
        raise RuntimeError("Invalid yfinance frame (empty or missing DatetimeIndex).")

    idx = frame.index.tz_localize(None)
    cutoff = pd.Timestamp(on_or_before) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    mask = idx <= cutoff
    if not bool(mask.any()):
        raise RuntimeError(f"No price data on or before {on_or_before.isoformat()} in downloaded frame.")

    sub = _get_field_frame(frame, field)
    last_dt = pd.Timestamp(idx[mask][-1]).tz_localize(None)

    if isinstance(frame.columns, pd.MultiIndex):
        prices: Dict[str, float] = {}
        for t in tickers:
            if t in sub.columns:
                vv = sub.loc[mask, t].dropna()
                prices[t] = float(vv.iloc[-1]) if len(vv) > 0 else np.nan
            else:
                prices[t] = np.nan
        return last_dt, pd.Series(prices, dtype="float64").reindex(tickers)

    vv = sub.loc[mask].iloc[:, 0].dropna()
    v = float(vv.iloc[-1]) if len(vv) > 0 else np.nan
    return last_dt, pd.Series({tickers[0]: v}, dtype="float64").reindex(tickers)


# ==========================
# REBALANCE STATE (forced)
# ==========================
def _default_last_rebalance_date() -> date:
    d = _parse_date(LAST_REBALANCE_DATE_STR)
    return d if d is not None else date.today()


def _compute_lockup_until(last_reb: date) -> date:
    if not ENFORCE_LOCKUP:
        return last_reb
    return last_reb + timedelta(days=int(LOCKUP_DAYS))


def load_rebalance_state() -> dict:
    return load_json(REBAL_JSON)


def normalize_rebalance_state() -> dict:
    """
    Ensures rebalance_state.json always reflects LAST_REBALANCE_DATE (forced by default)
    and lockup_until computed from that + LOCKUP_DAYS (when enforced).
    """
    st = load_rebalance_state() or {}
    forced = _default_last_rebalance_date()

    if FORCE_LAST_REBALANCE_DATE:
        last_reb = forced
        st["reason"] = "forced_last_rebalance_date"
    else:
        last_reb = _parse_date(st.get("last_rebalance_date")) or forced
        st["reason"] = st.get("reason") or "normalized"

    desired_lock_until = _compute_lockup_until(last_reb)

    st["last_rebalance_date"] = last_reb.isoformat()
    st["lockup_until"] = desired_lock_until.isoformat()
    st["lockup_days"] = int(LOCKUP_DAYS)
    st["lockup_enforced"] = bool(ENFORCE_LOCKUP)

    save_json(REBAL_JSON, st)
    return st


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
    st = normalize_rebalance_state()
    until = _parse_date(st.get("lockup_until"))
    if not ENFORCE_LOCKUP or until is None:
        return (False, until)
    return (as_of < until, until)


# ==========================
# HOLDINGS STATE (shares per yf_ticker)
# ==========================
def load_holdings() -> dict:
    return load_json(HOLDINGS_JSON)


def save_holdings(holdings: dict) -> None:
    save_json(HOLDINGS_JSON, holdings)


def _build_target_shares_at_anchor(port: pd.DataFrame, prices_at_anchor: pd.Series, anchor_date_used: date) -> dict:
    tickers = port["yf_ticker"].tolist()
    px = prices_at_anchor.reindex(tickers).astype(float)

    if px.isna().any() or (px <= 0).any():
        bad = px[px.isna() | (px <= 0)]
        raise RuntimeError(f"Cannot build anchor holdings; bad prices: {bad.to_dict()}")

    total = float(START_CAPITAL)
    tgt = port["target_w"].to_numpy(float)

    shares: Dict[str, float] = {}
    for yf_t, w, p in zip(tickers, tgt, px.to_numpy(float)):
        shares[yf_t] = float((total * float(w)) / float(p))

    return {
        "anchor_date": anchor_date_used.isoformat(),
        "capital_init": float(START_CAPITAL),
        "shares": shares,
        "note": "synthetic target shares set at anchor_date close (rebalance date)",
    }


def ensure_holdings_initialized(
    port: pd.DataFrame,
    prices_at_anchor: pd.Series,
    anchor_date_used: date,
) -> dict:
    """
    Ensures holdings_state.json exists and is anchored to the last rebalance date.
    If FORCE_HOLDINGS_TO_REBALANCE_DATE=1, it will rebuild shares whenever the stored anchor_date differs.
    """
    h = load_holdings()
    have_shares = isinstance(h.get("shares"), dict) and len(h.get("shares", {})) > 0
    stored_anchor = _parse_date(h.get("anchor_date"))

    if have_shares and stored_anchor and not FORCE_HOLDINGS_TO_REBALANCE_DATE and stored_anchor == anchor_date_used:
        return h

    if have_shares and stored_anchor and FORCE_HOLDINGS_TO_REBALANCE_DATE and stored_anchor == anchor_date_used:
        return h

    # Rebuild to anchor
    h = _build_target_shares_at_anchor(port, prices_at_anchor, anchor_date_used)
    save_holdings(h)
    _info(f"Holdings anchored to {anchor_date_used.isoformat()} using START_CAPITAL={START_CAPITAL:,.2f}")
    return h


# ==========================
# COMPUTE allocation
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


# ==========================
# REBALANCE execution (synthetic)
# ==========================
def rebalance_to_targets(port: pd.DataFrame, close_by_yf: pd.Series, total_value: float) -> Dict[str, float]:
    tickers = port["yf_ticker"].tolist()
    px = close_by_yf.reindex(tickers).astype(float)

    if px.isna().any() or (px <= 0).any():
        bad = px[px.isna() | (px <= 0)]
        raise RuntimeError(f"Cannot rebalance; bad prices: {bad.to_dict()}")

    tgt = port["target_w"].to_numpy(float)
    new_shares: Dict[str, float] = {}
    for yf_t, w, p in zip(tickers, tgt, px.to_numpy(float)):
        new_shares[yf_t] = float((float(total_value) * float(w)) / float(p))
    return new_shares


# ==========================
# SIGNALS STATE (with first_seen/last_seen)
# ==========================
def load_signals_state() -> dict:
    return load_json(SIGNALS_JSON)


def save_signals_state(state: dict) -> None:
    save_json(SIGNALS_JSON, state)


def current_signal_map(tbl: pd.DataFrame) -> dict:
    out = {}
    for _, r in tbl.iterrows():
        if r["status"] in ("LOW", "HIGH"):
            out[str(r["symbol"])] = str(r["status"])
    return out


def update_signal_timestamps(prev_state: dict, now_signals: dict, as_of: date) -> dict:
    first_seen = dict(prev_state.get("first_seen", {}) or {})
    last_seen = dict(prev_state.get("last_seen", {}) or {})

    for sym in now_signals.keys():
        if sym not in first_seen:
            first_seen[sym] = as_of.isoformat()
        last_seen[sym] = as_of.isoformat()

    return {
        "as_of": as_of.isoformat(),
        "signals": now_signals,
        "first_seen": first_seen,
        "last_seen": last_seen,
    }


# ==========================
# HISTORY — hardened
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
# Formatting helpers
# ==========================
def fmt_pct(x: float) -> str:
    return f"{x*100:6.2f}%"


def format_alloc_table(tbl: pd.DataFrame) -> str:
    lines = []
    lines.append(f"{'Symbol':<12} {'YF':<6} {'Now':>9} {'Target':>9} {'Band':>19} {'Status':>8} {'$Value':>14}")
    for _, r in tbl.iterrows():
        sym = str(r["symbol"])
        yf_t = str(r["yf_ticker"])
        now = fmt_pct(float(r["weight_now"]))
        tgt = fmt_pct(float(r["target_w"]))
        bd = float(r["band_down"])
        bu = float(r["band_up"])
        if bd <= 0 and bu <= 0:
            band = "—"
        else:
            band = f"[{r['lo']*100:5.2f}%..{r['hi']*100:5.2f}%]"
        st = str(r["status"])
        v = float(r["value"])
        lines.append(f"{sym:<12} {yf_t:<6} {now:>9} {tgt:>9} {band:>19} {st:>8} {v:>14,.2f}")
    return "\n".join(lines)


def rebalance_deltas(tbl: pd.DataFrame) -> pd.DataFrame:
    total = float(tbl["value"].sum())
    df = tbl.copy()
    df["target_$"] = df["target_w"] * total
    df["$delta_to_target"] = df["target_$"] - df["value"]
    return df[["symbol", "yf_ticker", "value", "weight_now", "target_w", "$delta_to_target"]].sort_values(
        "$delta_to_target", ascending=False
    )


def suggest_contribution_split(tbl: pd.DataFrame, contribution: float) -> pd.DataFrame:
    df = tbl.copy()
    df["deficit"] = np.maximum(0.0, df["target_w"] - df["weight_now"])

    if contribution <= 0:
        df["buy_$"] = 0.0
        return df[["symbol", "yf_ticker", "weight_now", "target_w", "deficit", "buy_$"]]

    d = df["deficit"].to_numpy(float)
    if float(d.sum()) <= 1e-12:
        w = df["target_w"].to_numpy(float)
        w = w / max(EPS, float(w.sum()))
    else:
        w = d / float(d.sum())

    df["buy_$"] = contribution * w
    return df[["symbol", "yf_ticker", "weight_now", "target_w", "deficit", "buy_$"]].sort_values("buy_$", ascending=False)


def price_change_table(
    port: pd.DataFrame,
    px_anchor: pd.Series,
    px_now: pd.Series,
) -> pd.DataFrame:
    df = port[["symbol", "yf_ticker", "target_w"]].copy()
    df["price_anchor"] = df["yf_ticker"].map(lambda t: float(px_anchor.get(t, np.nan)))
    df["price_now"] = df["yf_ticker"].map(lambda t: float(px_now.get(t, np.nan)))
    df["ret_since_rebalance"] = (df["price_now"] / df["price_anchor"] - 1.0).replace([np.inf, -np.inf], np.nan)
    return df


# ==========================
# JOBS
# ==========================
def daily_job() -> None:
    # Normalize rebalance state first (forces last_rebalance_date + lockup_until)
    reb_state = normalize_rebalance_state()
    last_reb = _parse_date(reb_state.get("last_rebalance_date")) or _default_last_rebalance_date()

    yf_tickers = PORT["yf_ticker"].tolist()

    # We need prices at the rebalance anchor date AND latest available close.
    # Download from a little before the rebalance date through "tomorrow" (end is exclusive).
    start = last_reb - timedelta(days=7)
    end = date.today() + timedelta(days=2)

    data = _yf_download(yf_tickers, start=start, end=end, interval="1d", tries=3)
    if not isinstance(data, pd.DataFrame) or len(data) == 0 or not isinstance(data.index, pd.DatetimeIndex):
        raise RuntimeError("yfinance returned no usable data; cannot run daily job.")

    # Latest available closes (as_of_dt can be earlier than calendar today)
    as_of_dt, close_now = _last_non_nan_prices(data, "Close", yf_tickers)
    as_of = as_of_dt.date()

    # Anchor prices (last trading day on/before last_reb)
    anchor_dt, close_anchor = _prices_on_or_before(data, "Close", yf_tickers, last_reb)
    anchor_used = anchor_dt.date()

    if close_now.isna().any():
        bad = close_now[close_now.isna()]
        raise RuntimeError(f"Missing Close prices for: {bad.index.tolist()}")

    if close_anchor.isna().any():
        bad = close_anchor[close_anchor.isna()]
        raise RuntimeError(f"Missing anchor Close prices for: {bad.index.tolist()} (anchor={anchor_used.isoformat()})")

    # Ensure holdings are anchored to rebalance date prices
    holdings = ensure_holdings_initialized(PORT, close_anchor, anchor_used)
    shares_by_yf = holdings.get("shares", {}) or {}

    # Compute today's allocation/value based on anchored shares and today's prices
    snap = compute_allocation(PORT, shares_by_yf, close_now, as_of)

    # Anchor portfolio value (should be ~= START_CAPITAL by construction; compute anyway)
    anchor_value = float(
        sum(float(shares_by_yf.get(t, 0.0)) * float(close_anchor.get(t, np.nan)) for t in yf_tickers)
    )
    since_reb_ret = (snap.total_value / anchor_value - 1.0) if np.isfinite(anchor_value) and anchor_value > 0 else np.nan

    roc_1d = update_history(as_of, snap.total_value)

    # Signals
    sig_now = current_signal_map(snap.table)
    prev_sig_state = load_signals_state()
    prev_sig_map = dict(prev_sig_state.get("signals", {}) or {})
    changed = (sig_now != prev_sig_map)

    new_sig_state = update_signal_timestamps(prev_sig_state, sig_now, as_of)
    save_signals_state(new_sig_state)

    # Lockup state
    in_lock, lock_until = lockup_status(as_of)

    did_rebalance = False
    reb_msg = ""

    # Optional auto-rebalance (if enabled) uses today's prices and resets anchor to as_of
    if sig_now and AUTO_REBALANCE:
        if ENFORCE_LOCKUP and in_lock:
            reb_msg = f"Auto-rebalance BLOCKED by lockup (until {lock_until.isoformat() if lock_until else '—'})."
        else:
            new_shares = rebalance_to_targets(PORT, close_now, snap.total_value)

            holdings = {
                "anchor_date": as_of.isoformat(),
                "capital_init": float(snap.total_value),
                "shares": new_shares,
                "note": "synthetic target shares set at auto-rebalance date close",
            }
            save_holdings(holdings)

            until = _compute_lockup_until(as_of) if ENFORCE_LOCKUP else as_of
            set_rebalance_state(
                last_rebalance_date=as_of,
                lockup_until=until,
                reason=f"auto_rebalance_on_breach (symbols={list(sig_now.keys())})",
            )

            did_rebalance = True
            reb_msg = f"Auto-rebalance EXECUTED. Lockup until {until.isoformat()}."

            # Recompute after rebalance
            snap = compute_allocation(PORT, new_shares, close_now, as_of)
            sig_now = current_signal_map(snap.table)

            prev_sig_state = load_signals_state()
            new_sig_state = update_signal_timestamps(prev_sig_state, sig_now, as_of)
            save_signals_state(new_sig_state)

            in_lock, lock_until = lockup_status(as_of)
            reb_state = load_rebalance_state()
            last_reb = _parse_date(reb_state.get("last_rebalance_date")) or as_of
            anchor_used = as_of
            anchor_value = float(snap.total_value)
            since_reb_ret = 0.0

    elif sig_now and not AUTO_REBALANCE:
        reb_msg = "Auto-rebalance OFF (AUTO_REBALANCE=0)."

    # status.json
    price_tbl = price_change_table(PORT, close_anchor, close_now)
    status = {
        "as_of": as_of.isoformat(),
        "as_of_timestamp": str(as_of_dt),
        "anchor_rebalance_date_requested": last_reb.isoformat(),
        "anchor_date_used": anchor_used.isoformat(),
        "anchor_value": float(anchor_value) if np.isfinite(anchor_value) else None,
        "total_value": float(snap.total_value),
        "return_since_rebalance": float(since_reb_ret) if np.isfinite(since_reb_ret) else None,
        "roc_1d": float(roc_1d),
        "last_rebalance_date": (last_reb.isoformat() if last_reb else None),
        "lockup_days": int(LOCKUP_DAYS),
        "lockup_enforced": bool(ENFORCE_LOCKUP),
        "lockup_until": (lock_until.isoformat() if lock_until else None),
        "in_lockup": bool(in_lock),
        "auto_rebalance": bool(AUTO_REBALANCE),
        "auto_rebalance_executed_today": bool(did_rebalance),
        "daily_email": bool(DAILY_EMAIL),
        "signals": sig_now,
        "signals_first_seen": new_sig_state.get("first_seen", {}),
        "signals_last_seen": new_sig_state.get("last_seen", {}),
        "portfolio": PORT.to_dict(orient="records"),
        "allocation": snap.table.to_dict(orient="records"),
        "prices": {
            "anchor_close": close_anchor.to_dict(),
            "latest_close": close_now.to_dict(),
        },
        "price_change_since_rebalance": price_tbl.to_dict(orient="records"),
    }
    Path(STATUS_JSON).write_text(json.dumps(status, indent=2), encoding="utf-8")

    breaches_exist = bool(sig_now)

    # Email decision:
    if DAILY_EMAIL:
        should_email = True
    else:
        should_email = breaches_exist and (ALWAYS_ALERT_ON_BREACH or changed or did_rebalance)

    # Email content
    header = [
        f"As of: {as_of.isoformat()} (EOD Close)",
        f"Rebalance date (requested): {last_reb.isoformat()}  | anchor used: {anchor_used.isoformat()}",
        f"Anchor value: ${anchor_value:,.2f}" if np.isfinite(anchor_value) else "Anchor value: n/a",
        f"Total value: ${snap.total_value:,.2f}   1d ROC: {roc_1d*100:+.2f}%",
        f"Return since rebalance: {since_reb_ret*100:+.2f}%" if np.isfinite(since_reb_ret) else "Return since rebalance: n/a",
        f"Lockup: {'ON' if ENFORCE_LOCKUP else 'OFF'} ({LOCKUP_DAYS} calendar days)",
        f"Locked until: {lock_until.isoformat() if lock_until else '—'}  (in_lockup={in_lock})",
        f"Auto-rebalance: {'ON' if AUTO_REBALANCE else 'OFF'}",
        f"Daily email: {'ON' if DAILY_EMAIL else 'OFF'}",
    ]
    if reb_msg:
        header.append(reb_msg)

    body = "\n".join(header) + "\n\n"
    body += "Allocation / Bands:\n"
    body += format_alloc_table(snap.table) + "\n\n"

    # Add price changes since rebalance
    pct = price_tbl.copy()
    pct["ret_since_rebalance"] = pct["ret_since_rebalance"].map(lambda x: f"{x*100:+.2f}%" if np.isfinite(x) else "n/a")
    body += "Price change since rebalance (anchor close → latest close):\n"
    for _, r in pct.iterrows():
        body += f"- {r['symbol']} ({r['yf_ticker']}): {r['ret_since_rebalance']}  ({r['price_anchor']:.2f} → {r['price_now']:.2f})\n"
    body += "\n"

    if breaches_exist:
        body += "⚠️ Band Breaches:\n"
        for sym, st in sig_now.items():
            r = snap.table[snap.table["symbol"] == sym].iloc[0]
            first_seen = new_sig_state.get("first_seen", {}).get(sym, "—")
            last_seen = new_sig_state.get("last_seen", {}).get(sym, "—")
            body += (
                f"- {sym} ({r['yf_ticker']}): {st}  "
                f"now={r['weight_now']*100:.2f}%  "
                f"band=[{r['lo']*100:.2f}%..{r['hi']*100:.2f}%]  "
                f"target={r['target_w']*100:.2f}%  "
                f"(first_seen={first_seen}, last_seen={last_seen})\n"
            )
        body += "\n"

    reb_df = rebalance_deltas(snap.table)
    contrib_df = suggest_contribution_split(snap.table, MONTHLY_CONTRIBUTION)

    if should_email:
        subj = "Portfolio — Daily update"
        if breaches_exist:
            subj = "Portfolio — Daily update (signals)"
        if did_rebalance:
            subj = "Portfolio — AUTO-REBALANCED (breach)"

        email_send(
            subject=subj,
            body=body,
            attachments=[
                ("allocation.csv", snap.table),
                ("rebalance_to_target_deltas.csv", reb_df),
                (f"contribution_split_${int(MONTHLY_CONTRIBUTION)}.csv", contrib_df),
                ("price_change_since_rebalance.csv", price_tbl),
            ],
        )
    else:
        _info("No email sent (DAILY_EMAIL=0 and no qualifying event).")


def weekly_job() -> None:
    """
    Weekly email is still supported (your workflow gates it to Fridays).
    It uses current anchored holdings and history.csv for performance ranges.
    """
    # Ensure rebalance state is normalized so lockup fields are correct
    reb_state = normalize_rebalance_state()
    last_reb = _parse_date(reb_state.get("last_rebalance_date")) or _default_last_rebalance_date()

    yf_tickers = PORT["yf_ticker"].tolist()

    # Pull enough to get latest closes
    data = _yf_download(yf_tickers, period="220d", interval="1d", tries=3)
    if not isinstance(data, pd.DataFrame) or len(data) == 0 or not isinstance(data.index, pd.DatetimeIndex):
        raise RuntimeError("yfinance returned no usable data; cannot run weekly job.")

    as_of_dt, close_now = _last_non_nan_prices(data, "Close", yf_tickers)
    as_of = as_of_dt.date()

    if close_now.isna().any():
        bad = close_now[close_now.isna()]
        raise RuntimeError(f"Missing Close prices for: {bad.index.tolist()}")

    holdings = load_holdings()
    shares_by_yf = holdings.get("shares", {}) or {}
    if not (isinstance(shares_by_yf, dict) and len(shares_by_yf) > 0):
        # If missing, anchor holdings using last rebalance date prices (best effort)
        start = last_reb - timedelta(days=7)
        end = date.today() + timedelta(days=2)
        data2 = _yf_download(yf_tickers, start=start, end=end, interval="1d", tries=3)
        anchor_dt, close_anchor = _prices_on_or_before(data2, "Close", yf_tickers, last_reb)
        holdings = ensure_holdings_initialized(PORT, close_anchor, anchor_dt.date())
        shares_by_yf = holdings.get("shares", {}) or {}

    snap = compute_allocation(PORT, shares_by_yf, close_now, as_of)

    in_lock, lock_until = lockup_status(as_of)

    # Performance from history.csv
    ret_5d = np.nan
    ret_mtd = np.nan
    ret_ytd = np.nan
    hist_df = pd.DataFrame(columns=["date", "total_value"])

    if Path(HISTORY_CSV).exists():
        try:
            raw_hist = pd.read_csv(HISTORY_CSV)
            hist = _collapse_history_df(raw_hist).sort_values("date")
            hist_df = hist.copy()

            if len(hist) >= 2:
                if len(hist) >= 6:
                    v0 = float(hist["total_value"].iloc[-6])
                    v1 = float(hist["total_value"].iloc[-1])
                    ret_5d = (v1 / v0 - 1.0) if v0 > 0 else np.nan

                this_month = hist[hist["date"].dt.to_period("M") == hist["date"].iloc[-1].to_period("M")]
                if len(this_month) >= 2:
                    v0 = float(this_month["total_value"].iloc[0])
                    v1 = float(this_month["total_value"].iloc[-1])
                    ret_mtd = (v1 / v0 - 1.0) if v0 > 0 else np.nan

                this_year = hist[hist["date"].dt.year == hist["date"].iloc[-1].year]
                if len(this_year) >= 2:
                    v0 = float(this_year["total_value"].iloc[0])
                    v1 = float(this_year["total_value"].iloc[-1])
                    ret_ytd = (v1 / v0 - 1.0) if v0 > 0 else np.nan
        except Exception as e:
            _warn("weekly: failed reading history.csv:", repr(e))

    sig_state = load_signals_state()
    sig_now = dict(sig_state.get("signals", {}) or {})
    first_seen = dict(sig_state.get("first_seen", {}) or {})
    last_seen = dict(sig_state.get("last_seen", {}) or {})

    header = [
        f"Weekly recap — as of {as_of.isoformat()} (EOD Close)",
        f"Total value: ${snap.total_value:,.2f}",
        f"Lockup: {'ON' if ENFORCE_LOCKUP else 'OFF'} ({LOCKUP_DAYS} calendar days)",
        f"Last rebalance (forced/normalized): {last_reb.isoformat()}",
        f"Locked until: {lock_until.isoformat() if lock_until else '—'}  (in_lockup={in_lock})",
        f"Auto-rebalance: {'ON' if AUTO_REBALANCE else 'OFF'}",
    ]

    body = "\n".join(header) + "\n\n"
    body += "Allocation / Bands:\n"
    body += format_alloc_table(snap.table) + "\n\n"

    body += "Performance (from history.csv total_value):\n"
    body += (f"- ~5d: {ret_5d*100:6.2f}%\n" if np.isfinite(ret_5d) else "- ~5d:   n/a\n")
    body += (f"- MTD: {ret_mtd*100:6.2f}%\n" if np.isfinite(ret_mtd) else "- MTD:  n/a\n")
    body += (f"- YTD: {ret_ytd*100:6.2f}%\n" if np.isfinite(ret_ytd) else "- YTD:  n/a\n")

    if sig_now:
        body += "\nOpen / recent signals:\n"
        for sym, st in sig_now.items():
            body += f"- {sym}: {st} (first_seen={first_seen.get(sym,'—')}, last_seen={last_seen.get(sym,'—')})\n"

    reb_df = rebalance_deltas(snap.table)
    contrib_df = suggest_contribution_split(snap.table, MONTHLY_CONTRIBUTION)

    email_send(
        subject="Portfolio — Weekly recap",
        body=body,
        attachments=[
            ("allocation.csv", snap.table),
            ("rebalance_to_target_deltas.csv", reb_df),
            (f"contribution_split_${int(MONTHLY_CONTRIBUTION)}.csv", contrib_df),
            ("history.csv", hist_df),
        ],
    )


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
            f"AUTO_REBALANCE={AUTO_REBALANCE}",
            f"ENFORCE_LOCKUP={ENFORCE_LOCKUP}",
            f"LOCKUP_DAYS={LOCKUP_DAYS}",
            f"LAST_REBALANCE_DATE={LAST_REBALANCE_DATE_STR}",
            f"FORCE_LAST_REBALANCE_DATE={FORCE_LAST_REBALANCE_DATE}",
            f"FORCE_HOLDINGS_TO_REBALANCE_DATE={FORCE_HOLDINGS_TO_REBALANCE_DATE}",
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
