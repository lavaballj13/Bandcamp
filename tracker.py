#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GitHub Actions Portfolio Tracker (bands + lockup + optional auto-rebalance + email)
HARDCODED PORTFOLIO (no portfolio.csv needed)

Key behavior:
- Prices come from yfinance using yf_ticker (e.g., TQQQ, DBMF, XLE, GLD, SGOV)
- Reporting/alerts use your internal symbol names (e.g., QQQSIM?L=3)
- Tracks synthetic fractional shares persisted in holdings_state.json
- Band breaches are evaluated on current weights vs target +/- (band_down, band_up)
- Lockup is CALENDAR DAYS (date difference in days)
- Optional AUTO_REBALANCE=1:
    If breach exists and not in lockup, rebalance to targets and set lockup_until.

Artifacts:
- holdings_state.json
- rebalance_state.json
- status.json
- history.csv
- signals_state.json

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
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf


# ==========================
# HARDCODED PORTFOLIO (your CSV baked in)
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

# normalize targets defensively (should already sum to 1)
PORT["target_w"] = pd.to_numeric(PORT["target_w"], errors="coerce").fillna(0.0)
s = float(PORT["target_w"].sum())
if s <= 0:
    raise ValueError("PORT target_w sums to 0; fix configuration.")
PORT["target_w"] = PORT["target_w"] / s

for c in ["band_down", "band_up"]:
    PORT[c] = pd.to_numeric(PORT[c], errors="coerce").fillna(0.0).clip(lower=0.0, upper=0.95)

PORT = PORT.reset_index(drop=True)


# ==========================
# ENV CONFIG
# ==========================
START_CAPITAL = float(os.getenv("START_CAPITAL", "100000.0"))

TAX_LOCKUP_DAYS = int(os.getenv("TAX_LOCKUP_DAYS", "365"))
ENFORCE_LOCKUP = (os.getenv("ENFORCE_LOCKUP", "1") == "1")

AUTO_REBALANCE = (os.getenv("AUTO_REBALANCE", "0") == "1")

ALWAYS_ALERT_ON_BREACH = (os.getenv("ALWAYS_ALERT_ON_BREACH", "0") == "1")

MONTHLY_CONTRIBUTION = float(os.getenv("MONTHLY_CONTRIBUTION", "2000.0"))

DBG = (os.getenv("TRACKER_DEBUG", "0") == "1")

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
# EMAIL (GitHub Secrets)
# ==========================
SMTP_HOST = os.getenv("SMTP_HOST")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASS = os.getenv("SMTP_PASS")
MAIL_FROM = os.getenv("MAIL_FROM")
MAIL_TO = os.getenv("MAIL_TO")
EMAIL_DISABLE = (os.getenv("EMAIL_DISABLE", "0") == "1")


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
        return json.loads(p.read_text())
    except Exception:
        return {}


def save_json(path: str, obj: dict) -> None:
    Path(path).write_text(json.dumps(obj, indent=2))


def _parse_date(s: str | None) -> date | None:
    if not s:
        return None
    try:
        return pd.to_datetime(s).date()
    except Exception:
        return None


# ==========================
# EMAIL
# ==========================
def email_send(subject: str, body: str, attachments: List[Tuple[str, pd.DataFrame]] | None = None) -> None:
    if EMAIL_DISABLE:
        _warn("EMAIL_DISABLE=1 → skipping email send")
        return
    if not all([SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS, MAIL_FROM, MAIL_TO]):
        _warn("Email not configured; set SMTP_* and MAIL_* secrets.")
        return

    try:
        msg = EmailMessage()
        msg["From"] = MAIL_FROM
        msg["To"] = MAIL_TO
        msg["Subject"] = subject
        msg.set_content(body)

        for name, df in (attachments or []):
            buf = io.StringIO()
            df.to_csv(buf, index=False)
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
def _yf_download(tickers: List[str], period="10d", interval="1d", tries=3) -> pd.DataFrame:
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
        time.sleep(1.0 + 0.5 * k)
    _warn(f"yfinance download empty after {tries} tries; err={last_err}")
    return pd.DataFrame()


def _get_field_frame(frame: pd.DataFrame, field: str) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or len(frame) == 0:
        raise KeyError("empty frame")
    if isinstance(frame.columns, pd.MultiIndex):
        if field in frame.columns.get_level_values(0):
            return frame[field]
        if field == "Adj Close" and "Close" in frame.columns.get_level_values(0):
            _warn("Adj Close missing; using Close as fallback")
            return frame["Close"]
        raise KeyError(f"{field} not found in fields: {sorted(set(frame.columns.get_level_values(0)))}")
    else:
        if field in frame.columns:
            return frame[[field]]
        if field == "Adj Close" and "Close" in frame.columns:
            _warn("Adj Close missing; using Close as fallback")
            return frame[["Close"]]
        raise KeyError(f"{field} not found in columns: {frame.columns.tolist()}")


def _last_row_series(frame: pd.DataFrame, field: str, tickers: List[str]) -> pd.Series:
    try:
        sub = _get_field_frame(frame, field)
    except Exception:
        return pd.Series({t: np.nan for t in tickers}, dtype="float64")

    if isinstance(frame.columns, pd.MultiIndex):
        s = sub.iloc[-1]
        s.index = s.index.astype(str)
        return pd.Series(s, dtype="float64").reindex(tickers)
    else:
        return pd.Series({tickers[0]: float(sub.iloc[-1, 0])}, dtype="float64")


# ==========================
# REBALANCE STATE
# ==========================
def load_rebalance_state() -> dict:
    return load_json(REBAL_JSON)


def set_rebalance_state(*, last_rebalance_date: date, lockup_until: date, reason: str) -> None:
    st = {
        "last_rebalance_date": last_rebalance_date.isoformat(),
        "lockup_until": lockup_until.isoformat(),
        "tax_lockup_days": int(TAX_LOCKUP_DAYS),
        "reason": str(reason),
    }
    save_json(REBAL_JSON, st)


def lockup_status(as_of: date) -> Tuple[bool, date | None]:
    st = load_rebalance_state()
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


def ensure_holdings_initialized(port: pd.DataFrame, last_close_by_yf: pd.Series, as_of: date) -> dict:
    h = load_holdings()
    if h.get("shares") and isinstance(h["shares"], dict) and len(h["shares"]) > 0:
        return h

    tickers = port["yf_ticker"].tolist()
    px = last_close_by_yf.reindex(tickers).astype(float)
    if px.isna().any() or (px <= 0).any():
        bad = px[px.isna() | (px <= 0)]
        raise RuntimeError(f"Cannot initialize holdings; bad prices: {bad.to_dict()}")

    total = float(START_CAPITAL)
    tgt = port["target_w"].to_numpy(float)

    shares = {}
    for yf_t, w, p in zip(tickers, tgt, px.to_numpy(float)):
        dollars = total * float(w)
        shares[yf_t] = float(dollars / float(p))

    h = {
        "as_of": as_of.isoformat(),
        "capital_init": float(START_CAPITAL),
        "shares": shares,  # keyed by yf_ticker
    }
    save_holdings(h)
    _info("Initialized holdings_state.json using START_CAPITAL =", START_CAPITAL)
    return h


# ==========================
# COMPUTE allocation (report by symbol)
# ==========================
@dataclass
class AllocationSnapshot:
    as_of: date
    total_value: float
    table: pd.DataFrame  # rows by symbol/yf_ticker


def compute_allocation(port: pd.DataFrame, shares_by_yf: Dict[str, float], last_close_by_yf: pd.Series, as_of: date) -> AllocationSnapshot:
    df = port.copy()
    df["shares"] = df["yf_ticker"].map(lambda t: float(shares_by_yf.get(t, 0.0)))
    df["price"] = df["yf_ticker"].map(lambda t: float(last_close_by_yf.get(t, np.nan)))
    df["value"] = df["shares"] * df["price"]
    df["value"] = df["value"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    total = float(df["value"].sum())
    df["weight_now"] = df["value"] / total if total > 0 else 0.0

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
def rebalance_to_targets(port: pd.DataFrame, last_close_by_yf: pd.Series, total_value: float) -> Dict[str, float]:
    tickers = port["yf_ticker"].tolist()
    px = last_close_by_yf.reindex(tickers).astype(float)
    if px.isna().any() or (px <= 0).any():
        bad = px[px.isna() | (px <= 0)]
        raise RuntimeError(f"Cannot rebalance; bad prices: {bad.to_dict()}")

    tgt = port["target_w"].to_numpy(float)
    new_shares: Dict[str, float] = {}
    for yf_t, w, p in zip(tickers, tgt, px.to_numpy(float)):
        dollars = float(total_value) * float(w)
        new_shares[yf_t] = float(dollars / float(p))
    return new_shares


# ==========================
# SIGNAL state
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


# ==========================
# HISTORY
# ==========================
def _collapse_history_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return df
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")
    g = df.groupby(df["date"].dt.date, as_index=False).last()
    g["date"] = pd.to_datetime(g["date"])
    return g


def update_history(as_of: date, total_value: float, roc_1d: float) -> None:
    cols = ["date", "total_value", "roc_1d"]
    day = as_of.isoformat()
    new_row = pd.DataFrame([{"date": day, "total_value": float(total_value), "roc_1d": float(roc_1d)}], columns=cols)

    path = Path(HISTORY_CSV)
    if path.exists():
        try:
            df = pd.read_csv(path, parse_dates=["date"])
        except Exception:
            df = pd.DataFrame(columns=cols)
        df = _collapse_history_df(df)
        if len(df):
            df = df[df["date"].dt.date.astype(str) != day]
        df = pd.concat([df, new_row], ignore_index=True)
    else:
        df = new_row

    df = df.sort_values("date")
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df.to_csv(path, index=False)


# ==========================
# Formatting / contribution
# ==========================
def fmt_pct(x: float) -> str:
    return f"{x*100:6.2f}%"


def format_alloc_table(tbl: pd.DataFrame) -> str:
    lines = []
    lines.append(f"{'Symbol':<12} {'YF':<6} {'Now':>9} {'Target':>9} {'Band':>17} {'Status':>8} {'$Value':>14}")
    for _, r in tbl.iterrows():
        sym = str(r["symbol"])
        yf = str(r["yf_ticker"])
        now = fmt_pct(float(r["weight_now"]))
        tgt = fmt_pct(float(r["target_w"]))
        bd = float(r["band_down"])
        bu = float(r["band_up"])
        if bd <= 0 and bu <= 0:
            band = "—"
        else:
            band = f"[{fmt_pct(float(r['lo']))}..{fmt_pct(float(r['hi']))}]"
        st = str(r["status"])
        v = float(r["value"])
        lines.append(f"{sym:<12} {yf:<6} {now:>9} {tgt:>9} {band:>17} {st:>8} {v:>14,.2f}")
    return "\n".join(lines)


def format_rebalance_deltas(tbl: pd.DataFrame) -> pd.DataFrame:
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
        # no underweights, fall back to target
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
    yf_tickers = PORT["yf_ticker"].tolist()
    data = _yf_download(yf_tickers, period="10d", interval="1d", tries=3)

    if not isinstance(data, pd.DataFrame) or len(data) == 0 or not isinstance(data.index, pd.DatetimeIndex):
        raise RuntimeError("yfinance returned no data; cannot run daily job.")

    as_of_dt = pd.Timestamp(data.index[-1]).tz_localize(None)
    as_of = as_of_dt.date()

    last_close_by_yf = _last_row_series(data, "Close", yf_tickers)
    if last_close_by_yf.isna().any():
        bad = last_close_by_yf[last_close_by_yf.isna()]
        raise RuntimeError(f"Missing Close prices for: {bad.index.tolist()}")

    holdings = ensure_holdings_initialized(PORT, last_close_by_yf, as_of)
    shares_by_yf = holdings.get("shares", {})

    snap = compute_allocation(PORT, shares_by_yf, last_close_by_yf, as_of)

    # 1-day ROC from history
    roc_1d = 0.0
    if Path(HISTORY_CSV).exists():
        try:
            hist = pd.read_csv(HISTORY_CSV)
            if len(hist) >= 1:
                prev_val = float(hist["total_value"].iloc[-1])
                roc_1d = (snap.total_value / prev_val - 1.0) if prev_val > 0 else 0.0
        except Exception:
            roc_1d = 0.0

    update_history(as_of, snap.total_value, roc_1d)

    sig_now = current_signal_map(snap.table)
    sig_prev = load_signals_state()
    changed = (sig_now != sig_prev)
    save_signals_state(sig_now)

    in_lock, lock_until = lockup_status(as_of)
    reb_state = load_rebalance_state()
    last_reb = _parse_date(reb_state.get("last_rebalance_date"))

    did_rebalance = False
    reb_msg = ""

    if len(sig_now) > 0 and AUTO_REBALANCE:
        if ENFORCE_LOCKUP and in_lock:
            reb_msg = f"Auto-rebalance BLOCKED by lockup (until {lock_until})."
        else:
            new_shares = rebalance_to_targets(PORT, last_close_by_yf, snap.total_value)
            holdings["shares"] = new_shares
            holdings["as_of"] = as_of.isoformat()
            save_holdings(holdings)

            until = as_of + timedelta(days=int(TAX_LOCKUP_DAYS)) if ENFORCE_LOCKUP else as_of
            set_rebalance_state(
                last_rebalance_date=as_of,
                lockup_until=until,
                reason=f"auto_rebalance_on_breach (symbols={list(sig_now.keys())})",
            )

            did_rebalance = True
            reb_msg = f"Auto-rebalance EXECUTED. Lockup until {until}."

            # recompute post-rebalance snapshot
            snap = compute_allocation(PORT, new_shares, last_close_by_yf, as_of)
            sig_now = current_signal_map(snap.table)
            in_lock, lock_until = lockup_status(as_of)
            reb_state = load_rebalance_state()
            last_reb = _parse_date(reb_state.get("last_rebalance_date"))

    elif len(sig_now) > 0 and not AUTO_REBALANCE:
        reb_msg = "Auto-rebalance OFF (AUTO_REBALANCE=0)."

    # status.json
    status = {
        "as_of": as_of.isoformat(),
        "total_value": float(snap.total_value),
        "roc_1d": float(roc_1d),
        "last_rebalance_date": (last_reb.isoformat() if last_reb else None),
        "lockup_days": int(TAX_LOCKUP_DAYS),
        "lockup_enforced": bool(ENFORCE_LOCKUP),
        "lockup_until": (lock_until.isoformat() if lock_until else None),
        "in_lockup": bool(in_lock),
        "auto_rebalance": bool(AUTO_REBALANCE),
        "auto_rebalance_executed_today": bool(did_rebalance),
        "signals": sig_now,
        "portfolio": PORT.to_dict(orient="records"),
        "allocation": snap.table.to_dict(orient="records"),
    }
    Path(STATUS_JSON).write_text(json.dumps(status, indent=2))

    breaches_exist = (len(sig_now) > 0)
    should_email = breaches_exist and (ALWAYS_ALERT_ON_BREACH or changed or did_rebalance)

    header = [
        f"As of: {as_of.isoformat()} (EOD Close)",
        f"Total value: ${snap.total_value:,.2f}   1d ROC: {roc_1d*100:+.2f}%",
        f"Lockup: {'ON' if ENFORCE_LOCKUP else 'OFF'} ({TAX_LOCKUP_DAYS} calendar days)",
        f"Last rebalance: {last_reb.isoformat() if last_reb else '—'}",
        f"Locked until: {lock_until.isoformat() if lock_until else '—'}  (in_lockup={in_lock})",
        f"Auto-rebalance: {'ON' if AUTO_REBALANCE else 'OFF'}",
    ]
    if reb_msg:
        header.append(reb_msg)

    body = "\n".join(header) + "\n\n"
    body += "Allocation / Bands:\n"
    body += format_alloc_table(snap.table) + "\n\n"

    if breaches_exist:
        body += "⚠️ Band Breaches:\n"
        for sym, st in sig_now.items():
            r = snap.table[snap.table["symbol"] == sym].iloc[0]
            body += (
                f"- {sym} ({r['yf_ticker']}): {st}  now={r['weight_now']*100:.2f}% "
                f"band=[{r['lo']*100:.2f}%..{r['hi']*100:.2f}%] target={r['target_w']*100:.2f}%\n"
            )
        body += "\n"

    reb_df = format_rebalance_deltas(snap.table)
    contrib_df = suggest_contribution_split(snap.table, MONTHLY_CONTRIBUTION)

    if should_email:
        subj = "Portfolio — SIGNALS (bands breached)"
        if did_rebalance:
            subj = "Portfolio — AUTO-REBALANCED (breach)"
        email_send(
            subject=subj,
            body=body,
            attachments=[
                ("allocation.csv", snap.table),
                ("rebalance_to_target_deltas.csv", reb_df),
                ("contribution_split_$2000.csv", contrib_df),
            ],
        )
    else:
        _info("No email sent (no breaches or unchanged, and no auto-rebalance).")


# ==========================
# WEEKLY JOB
# ==========================
def weekly_job() -> None:
    yf_tickers = PORT["yf_ticker"].tolist()
    data = _yf_download(yf_tickers, period="30d", interval="1d", tries=3)

    if not isinstance(data, pd.DataFrame) or len(data) == 0 or not isinstance(data.index, pd.DatetimeIndex):
        raise RuntimeError("yfinance returned no data; cannot run weekly job.")

    as_of_dt = pd.Timestamp(data.index[-1]).tz_localize(None)
    as_of = as_of_dt.date()

    last_close_by_yf = _last_row_series(data, "Close", yf_tickers)
    if last_close_by_yf.isna().any():
        bad = last_close_by_yf[last_close_by_yf.isna()]
        raise RuntimeError(f"Missing Close prices for: {bad.index.tolist()}")

    holdings = ensure_holdings_initialized(PORT, last_close_by_yf, as_of)
    shares_by_yf = holdings.get("shares", {})

    snap = compute_allocation(PORT, shares_by_yf, last_close_by_yf, as_of)

    in_lock, lock_until = lockup_status(as_of)
    reb_state = load_rebalance_state()
    last_reb = _parse_date(reb_state.get("last_rebalance_date"))

    # performance from history.csv
    ret_5d = np.nan
    ret_mtd = np.nan
    ret_ytd = np.nan
    if Path(HISTORY_CSV).exists():
        try:
            hist = pd.read_csv(HISTORY_CSV)
            hist["date"] = pd.to_datetime(hist["date"])
            hist = hist.sort_values("date")
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

    header = [
        f"Weekly recap — as of {as_of.isoformat()} (EOD Close)",
        f"Total value: ${snap.total_value:,.2f}",
        f"Lockup: {'ON' if ENFORCE_LOCKUP else 'OFF'} ({TAX_LOCKUP_DAYS} calendar days)",
        f"Last rebalance: {last_reb.isoformat() if last_reb else '—'}",
        f"Locked until: {lock_until.isoformat() if lock_until else '—'}  (in_lockup={in_lock})",
        f"Auto-rebalance: {'ON' if AUTO_REBALANCE else 'OFF'}",
    ]
    if reb_state.get("reason"):
        header.append(f"Last rebalance reason: {reb_state.get('reason')}")

    body = "\n".join(header) + "\n\n"
    body += "Allocation / Bands:\n"
    body += format_alloc_table(snap.table) + "\n\n"

    body += "Performance (from history.csv total_value):\n"
    body += f"- ~5d: {ret_5d*100:6.2f}%\n" if np.isfinite(ret_5d) else "- ~5d:   n/a\n"
    body += f"- MTD: {ret_mtd*100:6.2f}%\n" if np.isfinite(ret_mtd) else "- MTD:  n/a\n"
    body += f"- YTD: {ret_ytd*100:6.2f}%\n" if np.isfinite(ret_ytd) else "- YTD:  n/a\n"

    reb_df = format_rebalance_deltas(snap.table)
    contrib_df = suggest_contribution_split(snap.table, MONTHLY_CONTRIBUTION)

    email_send(
        subject="Portfolio — Weekly recap",
        body=body,
        attachments=[
            ("allocation.csv", snap.table),
            ("rebalance_to_target_deltas.csv", reb_df),
            ("contribution_split_$2000.csv", contrib_df),
            ("history.csv", pd.read_csv(HISTORY_CSV) if Path(HISTORY_CSV).exists() else pd.DataFrame()),
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
