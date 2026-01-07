#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GitHub Actions Portfolio Tracker (bands + lockup + optional auto-rebalance + email)
HARD-CODED PORTFOLIO (no portfolio.csv needed)

Behavior:
- Anchors synthetic holdings to a configured rebalance date (default 2025-04-16):
    shares at anchor = START_CAPITAL * target_w / anchor_close
  Those shares are then valued at latest close to compute current weights/value.
- Lockup uses the anchor rebalance date:
    lockup_until = rebalance_date + LOCKUP_DAYS
- Daily email can be forced ON every run with DAILY_EMAIL=1 (default ON).
- Email body excludes anchor/total/1d ROC lines (per request).

Artifacts:
- holdings_state.json     (anchor_date, shares by yf_ticker)
- rebalance_state.json    (last_rebalance_date=anchor, lockup_until=anchor+LOCKUP_DAYS)
- signals_state.json      (current signals + first_seen/last_seen)
- status.json             (debug snapshot includes anchor info + totals)
- history.csv             (daily total_value; schema-hardened)

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
    d = pd.to_datetime(v, errors="raise").date()
    return d


# ==========================
# CONFIG
# ==========================
START_CAPITAL = _env_float("START_CAPITAL", "100000.0")

LOCKUP_DAYS = _env_int("LOCKUP_DAYS", os.getenv("TAX_LOCKUP_DAYS", "365"))
ENFORCE_LOCKUP = _env_bool("ENFORCE_LOCKUP", "1")

AUTO_REBALANCE = _env_bool("AUTO_REBALANCE", "0")
ALWAYS_ALERT_ON_BREACH = _env_bool("ALWAYS_ALERT_ON_BREACH", "0")
MONTHLY_CONTRIBUTION = _env_float("MONTHLY_CONTRIBUTION", "2000.0")

# Daily email every run (default ON)
DAILY_EMAIL = _env_bool("DAILY_EMAIL", "1")

DBG = _env_bool("TRACKER_DEBUG", "0")

# Rebalance anchor date (requested)
REBALANCE_DATE = _env_date("REBALANCE_DATE", "2025-04-16")

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
SMTP_PORT = _env_int("SMTP_PORT", "587")  # handles empty string safely
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
def _yf_download(tickers: List[str], period: str = "2y", interval: str = "1d", tries: int = 3) -> pd.DataFrame:
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
    df = df.sort_index()
    df = df.reindex(columns=tickers)

    df = df[df.index.date <= anchor_req]
    df = df.dropna(how="any")
    if len(df) == 0:
        raise RuntimeError(f"No common anchor close available on/before {anchor_req.isoformat()} for all tickers.")

    row = df.iloc[-1]
    anchor_used = df.index[-1].date()
    px = row.astype(float)
    return anchor_used, px


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


def ensure_holdings_anchored(
    port: pd.DataFrame,
    anchor_used: date,
    anchor_close_by_yf: pd.Series,
) -> dict:
    """
    Ensure holdings_state.json represents anchor-date synthetic shares computed
    from START_CAPITAL and target weights.

    If file exists but anchor_date differs, it will be overwritten to match.
    """
    h = load_holdings()
    existing_anchor = _parse_date(h.get("anchor_date") or h.get("as_of"))

    tickers = port["yf_ticker"].tolist()
    px = anchor_close_by_yf.reindex(tickers).astype(float)

    if px.isna().any() or (px <= 0).any():
        bad = px[px.isna() | (px <= 0)]
        raise RuntimeError(f"Cannot anchor holdings; bad anchor prices: {bad.to_dict()}")

    need_reset = True
    if isinstance(h.get("shares"), dict) and len(h["shares"]) > 0 and existing_anchor == anchor_used:
        need_reset = False

    if not need_reset:
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

    # Also anchor rebalance_state.json to the same anchor date
    lock_until = anchor_used + timedelta(days=int(LOCKUP_DAYS)) if ENFORCE_LOCKUP else anchor_used
    set_rebalance_state(
        last_rebalance_date=anchor_used,
        lockup_until=lock_until,
        reason="anchor_init",
    )
    return h_new


# ==========================
# ALLOCATION
# ==========================
@dataclass
class AllocationSnapshot:
    as_of: date
    total_value: float
    table: pd.DataFrame


def compute_allocation(
    port: pd.DataFrame,
    shares_by_yf: Dict[str, float],
    close_by_yf: pd.Series,
    as_of: date,
) -> AllocationSnapshot:
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


def current_signal_map(tbl: pd.DataFrame) -> dict:
    out = {}
    for _, r in tbl.iterrows():
        if r["status"] in ("LOW", "HIGH"):
            out[str(r["symbol"])] = str(r["status"])
    return out


# ==========================
# SIGNALS STATE (timestamps)
# ==========================
def load_signals_state() -> dict:
    return load_json(SIGNALS_JSON)


def save_signals_state(state: dict) -> None:
    save_json(SIGNALS_JSON, state)


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


# ==========================
# DAILY JOB
# ==========================
def daily_job() -> None:
    tickers = PORT["yf_ticker"].tolist()

    data = _yf_download(tickers, period="2y", interval="1d", tries=3)
    if not isinstance(data, pd.DataFrame) or len(data) == 0 or not isinstance(data.index, pd.DatetimeIndex):
        raise RuntimeError("yfinance returned no usable data; cannot run daily job.")

    close_df = _get_field_frame(data, "Close")
    close_df.index = pd.to_datetime(close_df.index).tz_localize(None)

    # Anchor prices (common date across all tickers)
    anchor_used, anchor_px = _anchor_prices_all_tickers(close_df, REBALANCE_DATE, tickers)

    # Latest close (per-ticker last non-NaN)
    as_of_dt, latest_px = _last_non_nan_prices(data, "Close", tickers)
    as_of = as_of_dt.date()
    if latest_px.isna().any():
        bad = latest_px[latest_px.isna()]
        raise RuntimeError(f"Missing latest Close prices for: {bad.index.tolist()}")

    # Anchor holdings + lockup state pinned to anchor_used
    holdings = ensure_holdings_anchored(PORT, anchor_used, anchor_px)
    shares_by_yf = holdings.get("shares", {}) or {}

    # Current allocation from anchor shares -> latest prices
    snap = compute_allocation(PORT, shares_by_yf, latest_px, as_of)

    # Return since rebalance (anchor capital to current total)
    anchor_value = float(START_CAPITAL)
    total_value = float(snap.total_value)
    ret_since = (total_value / anchor_value - 1.0) if anchor_value > 0 else np.nan

    # History (still maintained; roc not shown in email per request)
    roc_1d = update_history(as_of, total_value)

    # Signals
    sig_now = current_signal_map(snap.table)
    prev_sig_state = load_signals_state()
    prev_sig_map = dict(prev_sig_state.get("signals", {}) or {})
    changed = (sig_now != prev_sig_map)
    new_sig_state = update_signal_timestamps(prev_sig_state, sig_now, as_of)
    save_signals_state(new_sig_state)

    # Lockup (from rebalance_state.json which we pinned to anchor_used)
    in_lock, lock_until = lockup_status(as_of)
    reb_state = load_rebalance_state()
    last_reb = _parse_date(reb_state.get("last_rebalance_date"))

    did_rebalance = False
    reb_msg = ""
    # (Auto-rebalance would be based on breaches; OFF in your workflow)
    if sig_now and AUTO_REBALANCE:
        if ENFORCE_LOCKUP and in_lock:
            reb_msg = f"Auto-rebalance BLOCKED by lockup (until {lock_until.isoformat() if lock_until else '—'})."
        else:
            # Rebalance to exact targets at latest prices
            px = latest_px.reindex(tickers).astype(float)
            if px.isna().any() or (px <= 0).any():
                bad = px[px.isna() | (px <= 0)]
                raise RuntimeError(f"Cannot rebalance; bad prices: {bad.to_dict()}")

            tgt = PORT["target_w"].to_numpy(float)
            new_shares: Dict[str, float] = {}
            for yf_t, w, p in zip(tickers, tgt, px.to_numpy(float)):
                new_shares[yf_t] = float((total_value * float(w)) / float(p))

            holdings["shares"] = new_shares
            holdings["anchor_date"] = anchor_used.isoformat()
            save_holdings(holdings)

            until = anchor_used + timedelta(days=int(LOCKUP_DAYS)) if ENFORCE_LOCKUP else anchor_used
            set_rebalance_state(
                last_rebalance_date=anchor_used,
                lockup_until=until,
                reason=f"auto_rebalance_on_breach (symbols={list(sig_now.keys())})",
            )

            did_rebalance = True
            reb_msg = f"Auto-rebalance EXECUTED. Lockup until {until.isoformat()}."

            # Recompute post-rebalance
            snap = compute_allocation(PORT, new_shares, latest_px, as_of)
            sig_now = current_signal_map(snap.table)
            new_sig_state = update_signal_timestamps(load_signals_state(), sig_now, as_of)
            save_signals_state(new_sig_state)
            in_lock, lock_until = lockup_status(as_of)
            reb_state = load_rebalance_state()
            last_reb = _parse_date(reb_state.get("last_rebalance_date"))

    # status.json for debugging (can include the removed lines; email will not)
    status = {
        "as_of": as_of.isoformat(),
        "as_of_timestamp": str(as_of_dt),
        "rebalance_date_requested": REBALANCE_DATE.isoformat(),
        "anchor_used": anchor_used.isoformat(),
        "anchor_value": float(anchor_value),
        "total_value": float(snap.total_value),
        "roc_1d": float(roc_1d),
        "return_since_rebalance": float(ret_since) if np.isfinite(ret_since) else None,
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
        "anchor_prices": anchor_px.to_dict(),
        "latest_prices": latest_px.to_dict(),
    }
    Path(STATUS_JSON).write_text(json.dumps(status, indent=2), encoding="utf-8")

    # ---- EMAIL BODY (EXCLUDES: rebalance date line, anchor value line, total value + 1d ROC line) ----
    header = [
        f"As of: {as_of.isoformat()} (EOD Close)",
        (f"Return since rebalance: {ret_since*100:+.2f}%" if np.isfinite(ret_since) else "Return since rebalance: n/a"),
        f"Lockup: {'ON' if ENFORCE_LOCKUP else 'OFF'} ({LOCKUP_DAYS} calendar days)",
        f"Locked until: {lock_until.isoformat() if lock_until else '—'}  (in_lockup={in_lock})",
        f"Auto-rebalance: {'ON' if AUTO_REBALANCE else 'OFF'}",
        f"Daily email: ON",
    ]
    if reb_msg:
        header.append(reb_msg)

    body = "\n".join(header) + "\n\n"
    body += "Allocation / Bands:\n"
    body += format_alloc_table(snap.table) + "\n\n"

    # Keep price-change section (still useful)
    body += "Price change since rebalance (anchor close \u2192 latest close):\n"
    for sym, yf_t in zip(PORT["symbol"].tolist(), tickers):
        a = float(anchor_px.get(yf_t, np.nan))
        b = float(latest_px.get(yf_t, np.nan))
        if np.isfinite(a) and a > 0 and np.isfinite(b):
            pct = (b / a - 1.0) * 100.0
            body += f"- {sym} ({yf_t}): {pct:+.2f}%  ({a:.2f} \u2192 {b:.2f})\n"
        else:
            body += f"- {sym} ({yf_t}): n/a\n"

    # Optional breach section (fine to keep; it will often be empty)
    if sig_now:
        body += "\n\u26A0\uFE0F Band Breaches:\n"
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

    reb_df = rebalance_deltas(snap.table)
    contrib_df = suggest_contribution_split(snap.table, MONTHLY_CONTRIBUTION)

    # Email decision: daily always, plus special subjects if needed
    breaches_exist = bool(sig_now)
    should_email = bool(DAILY_EMAIL) or (breaches_exist and (ALWAYS_ALERT_ON_BREACH or changed or did_rebalance))

    if should_email:
        if did_rebalance:
            subj = "Portfolio — AUTO-REBALANCED"
        elif breaches_exist:
            subj = "Portfolio — Daily update (signals)"
        else:
            subj = "Portfolio — Daily update"

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
        _info("No email sent (daily email OFF and no breach-trigger).")


# ==========================
# WEEKLY JOB
# ==========================
def weekly_job() -> None:
    # Weekly just reuses daily_job-style logic but always emails a recap.
    tickers = PORT["yf_ticker"].tolist()
    data = _yf_download(tickers, period="2y", interval="1d", tries=3)
    if not isinstance(data, pd.DataFrame) or len(data) == 0 or not isinstance(data.index, pd.DatetimeIndex):
        raise RuntimeError("yfinance returned no usable data; cannot run weekly job.")

    close_df = _get_field_frame(data, "Close")
    close_df.index = pd.to_datetime(close_df.index).tz_localize(None)

    anchor_used, anchor_px = _anchor_prices_all_tickers(close_df, REBALANCE_DATE, tickers)
    as_of_dt, latest_px = _last_non_nan_prices(data, "Close", tickers)
    as_of = as_of_dt.date()

    holdings = ensure_holdings_anchored(PORT, anchor_used, anchor_px)
    shares_by_yf = holdings.get("shares", {}) or {}

    snap = compute_allocation(PORT, shares_by_yf, latest_px, as_of)

    # history-based perf (optional)
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

    in_lock, lock_until = lockup_status(as_of)
    reb_state = load_rebalance_state()
    last_reb = _parse_date(reb_state.get("last_rebalance_date"))

    header = [
        f"Weekly recap — as of {as_of.isoformat()} (EOD Close)",
        f"Lockup: {'ON' if ENFORCE_LOCKUP else 'OFF'} ({LOCKUP_DAYS} calendar days)",
        f"Last rebalance: {last_reb.isoformat() if last_reb else '—'}",
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
            f"ALWAYS_ALERT_ON_BREACH={ALWAYS_ALERT_ON_BREACH}",
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
