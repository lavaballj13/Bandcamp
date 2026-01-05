#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Band-based portfolio messenger (GitHub Actions friendly)

- Reads portfolio.csv:
    symbol,yf_ticker,target_w,band_down,band_up
  Bands are absolute deviations from target (like your optimizer).

- Builds baseline.json on first run (baseline Close & Adj Close).
- Daily EOD:
    - pulls latest prices (yfinance)
    - computes current weights from baseline-relative drift
    - checks band breaches
    - applies "rebalance lockup" (calendar days) after last recorded rebalance
    - emails on state change (or always if ALWAYS_ALERT_ON_BREACH=1)
    - writes artifacts & commits via workflow

- Weekly:
    - recap email with allocation and simple history-based returns

Rebalance tracking:
- rebalance_state.json stores:
    last_rebalance_date (YYYY-MM-DD)
    lockup_days
    locked_until (YYYY-MM-DD)
- You can mark a rebalance you actually executed:
    python tracker.py mark_rebalanced
  (This sets last_rebalance_date to latest market date in status, and locks until +LOCKUP_DAYS.)
"""

import os, io, json, smtplib, ssl, sys, time
from email.message import EmailMessage
from pathlib import Path
import math

import numpy as np
import pandas as pd
import yfinance as yf

# ---------- Env / knobs ----------
NOTIONAL_CAPITAL = float(os.getenv("NOTIONAL_CAPITAL", "100000.0"))

BASELINE_DATE = os.getenv("BASELINE_DATE", "")  # "" => use latest bar on first run

LOCKUP_DAYS = int(os.getenv("LOCKUP_DAYS", "365"))              # calendar days
ENFORCE_LOCKUP = (os.getenv("ENFORCE_LOCKUP", "1") == "1")      # suppress alerts during lockup
ALWAYS_ALERT_ON_BREACH = (os.getenv("ALWAYS_ALERT_ON_BREACH", "0") == "1")

DBG = (os.getenv("TRACKER_DEBUG", "0") == "1")
EMAIL_DISABLE = (os.getenv("EMAIL_DISABLE", "0") == "1")

# ---------- Files ----------
PORTFOLIO_CSV = "portfolio.csv"
BASELINE_JSON = "baseline.json"
STATUS_JSON   = "status.json"
HISTORY_CSV   = "history.csv"
SIGNALS_JSON  = "signals_state.json"
REBAL_STATE_JSON = "rebalance_state.json"

# ---------- Email secrets ----------
SMTP_HOST = os.getenv("SMTP_HOST")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASS = os.getenv("SMTP_PASS")
MAIL_FROM = os.getenv("MAIL_FROM")
MAIL_TO   = os.getenv("MAIL_TO")  # comma-separated ok

EPS = 1e-12

def _dbg(*a):
    if DBG:
        print("[dbg]", *a, flush=True)

def _warn(*a):
    print("[warn]", *a, flush=True)

def _err(*a):
    print("[error]", *a, flush=True)

# ============================ IO helpers ============================

def load_portfolio(path=PORTFOLIO_CSV) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = {"symbol","yf_ticker","target_w","band_down","band_up"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")

    df = df.copy()
    df["symbol"] = df["symbol"].astype(str)
    df["yf_ticker"] = df["yf_ticker"].fillna("").astype(str)

    for c in ["target_w","band_down","band_up"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    # normalize weights if needed
    s = float(df["target_w"].sum())
    if s > 0 and not np.isclose(s, 1.0):
        df["target_w"] = df["target_w"] / s

    # compute absolute band limits
    df["band_lo"] = (df["target_w"] - df["band_down"]).clip(lower=0.0, upper=1.0)
    df["band_hi"] = (df["target_w"] + df["band_up"]).clip(lower=0.0, upper=1.0)

    return df

def _read_json(path: str, default):
    p = Path(path)
    if not p.exists():
        return default
    try:
        return json.loads(p.read_text())
    except Exception:
        return default

def _write_json(path: str, obj) -> None:
    Path(path).write_text(json.dumps(obj, indent=2))

def _collapse_history_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return df
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")
    g = df.groupby(df["date"].dt.date, as_index=False).last()
    g["date"] = pd.to_datetime(g["date"])
    return g

def load_history(path=HISTORY_CSV) -> pd.DataFrame | None:
    if not Path(path).exists():
        return None
    df = pd.read_csv(path, parse_dates=["date"])
    return _collapse_history_df(df)

def upsert_history(day: str, mv: float, roc: float) -> None:
    cols = ["date", "mv", "roc"]
    new = pd.DataFrame([{"date": day, "mv": float(mv), "roc": float(roc)}], columns=cols)

    p = Path(HISTORY_CSV)
    if p.exists():
        df = pd.read_csv(p, parse_dates=["date"])
        df = _collapse_history_df(df)
        df = df[df["date"].dt.date.astype(str) != day]
        df = pd.concat([df, new], ignore_index=True)
    else:
        df = new

    df = df.sort_values("date")
    df_out = df.copy()
    df_out["date"] = df_out["date"].dt.strftime("%Y-%m-%d")
    df_out.to_csv(p, index=False)

# ============================ yfinance helpers ============================

def _yf_download(tickers, period="14d", interval="1d", tries=3) -> pd.DataFrame:
    tickers = [t for t in tickers if t]
    if not tickers:
        return pd.DataFrame()
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
        time.sleep(1.0 + 0.5*k)
    _warn(f"yfinance download empty after {tries} tries: {last_err}")
    return pd.DataFrame()

def _get_field(frame: pd.DataFrame, field: str) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or len(frame) == 0:
        raise KeyError("empty")
    if isinstance(frame.columns, pd.MultiIndex):
        if field in frame.columns.get_level_values(0):
            return frame[field]
        if field == "Adj Close" and "Close" in frame.columns.get_level_values(0):
            _warn("Adj Close missing; falling back to Close")
            return frame["Close"]
        raise KeyError(f"missing field={field}")
    else:
        if field in frame.columns:
            return frame[[field]]
        if field == "Adj Close" and "Close" in frame.columns:
            _warn("Adj Close missing; falling back to Close")
            return frame[["Close"]]
        raise KeyError(f"missing field={field}")

def _last_row_series(frame: pd.DataFrame, field: str, tickers: list[str]) -> pd.Series:
    if not isinstance(frame, pd.DataFrame) or len(frame) == 0:
        return pd.Series({t: np.nan for t in tickers}, dtype="float64")
    sub = _get_field(frame, field)
    if isinstance(frame.columns, pd.MultiIndex):
        s = sub.iloc[-1]
        s.index = s.index.astype(str)
        return pd.Series(s, dtype="float64").reindex(tickers)
    else:
        # single ticker
        return pd.Series({tickers[0]: float(sub.iloc[-1, 0])}, dtype="float64")

# ============================ baseline ============================

def load_or_create_baseline(yf_tickers: list[str]) -> dict:
    if Path(BASELINE_JSON).exists():
        js = _read_json(BASELINE_JSON, None)
        if isinstance(js, dict) and js.get("base_close") and js.get("as_of"):
            return js

    # create
    if BASELINE_DATE.strip():
        hist = yf.download(
            tickers=yf_tickers,
            start=BASELINE_DATE.strip(),
            interval="1d",
            auto_adjust=False,
            progress=False,
            group_by="column",
        )
        if not isinstance(hist, pd.DataFrame) or len(hist) == 0:
            # fallback recent
            hist = _yf_download(yf_tickers, period="60d", interval="1d", tries=3)
    else:
        hist = _yf_download(yf_tickers, period="14d", interval="1d", tries=3)

    if not isinstance(hist, pd.DataFrame) or len(hist) == 0:
        raise RuntimeError("Could not fetch data to create baseline")

    base_dt = pd.Timestamp(hist.index[0]).tz_localize(None)
    close_df = _get_field(hist, "Close")
    try:
        adj_df = _get_field(hist, "Adj Close")
    except Exception:
        adj_df = close_df

    if isinstance(hist.columns, pd.MultiIndex):
        base_close = close_df.iloc[0]; base_close.index = base_close.index.astype(str)
        base_adj = adj_df.iloc[0]; base_adj.index = base_adj.index.astype(str)
        base_close = pd.Series(base_close, dtype="float64").reindex(yf_tickers)
        base_adj = pd.Series(base_adj, dtype="float64").reindex(yf_tickers)
    else:
        base_close = pd.Series({yf_tickers[0]: float(close_df.iloc[0, 0])}, dtype="float64")
        base_adj = pd.Series({yf_tickers[0]: float(adj_df.iloc[0, 0])}, dtype="float64")

    baseline = {
        "as_of": str(base_dt.date()),
        "capital": NOTIONAL_CAPITAL,
        "base_close": {t: float(base_close.get(t, np.nan)) for t in yf_tickers},
        "base_adj": {t: float(base_adj.get(t, np.nan)) for t in yf_tickers},
    }
    _write_json(BASELINE_JSON, baseline)
    return baseline

# ============================ lockup / rebalance state ============================

def load_rebalance_state() -> dict:
    st = _read_json(REBAL_STATE_JSON, {})
    if not isinstance(st, dict):
        st = {}
    # normalize fields
    st.setdefault("lockup_days", LOCKUP_DAYS)
    st.setdefault("last_rebalance_date", "")
    st.setdefault("locked_until", "")
    return st

def save_rebalance_state(st: dict) -> None:
    _write_json(REBAL_STATE_JSON, st)

def lockup_status(asof_date: str, st: dict) -> tuple[bool, int]:
    """
    Returns (in_lockup, days_remaining) using calendar days.
    """
    if not ENFORCE_LOCKUP:
        return False, 0
    locked_until = st.get("locked_until", "")
    if not locked_until:
        return False, 0
    try:
        asof = pd.to_datetime(asof_date)
        until = pd.to_datetime(locked_until)
        if asof < until:
            return True, int((until - asof).days)
        return False, 0
    except Exception:
        return False, 0

# ============================ portfolio math ============================

def compute_allocation(port: pd.DataFrame, last_close_by_yf: pd.Series, baseline: dict) -> tuple[pd.DataFrame, float]:
    """
    Uses baseline-relative price drift to estimate weights:
      value_i = NOTIONAL * target_w_i * (price_i / base_price_i)
    If yf_ticker is blank => use drift=1.0 (constant).
    """
    df = port.copy()
    yf_tickers = df["yf_ticker"].tolist()

    base_close = baseline["base_close"]

    price = []
    drift = []
    for yf_t in yf_tickers:
        if not yf_t:
            price.append(np.nan)
            drift.append(1.0)
            continue
        p = float(last_close_by_yf.get(yf_t, np.nan))
        b = float(base_close.get(yf_t, np.nan))
        price.append(p)
        if (not np.isfinite(p)) or (not np.isfinite(b)) or b <= 0:
            drift.append(0.0)
        else:
            drift.append(max(0.0, p / b))

    df["price"] = price
    df["drift"] = drift

    target = df["target_w"].to_numpy(float)
    driftv = df["drift"].to_numpy(float)
    mv = NOTIONAL_CAPITAL * target * driftv
    total = float(np.nansum(mv))
    wnow = (mv / total) if total > 0 else np.zeros_like(mv)

    df["mkt_value"] = mv
    df["weight_now"] = wnow
    df["delta_w"] = df["weight_now"] - df["target_w"]

    def _status(row):
        w = float(row["weight_now"])
        lo = float(row["band_lo"])
        hi = float(row["band_hi"])
        if w < lo - EPS:
            return "LOW"
        if w > hi + EPS:
            return "HIGH"
        return "OK"

    df["status"] = df.apply(_status, axis=1)
    df["would_trigger_now"] = df["status"].isin(["LOW", "HIGH"])
    return df, total

def make_rebalance_deltas(df_alloc: pd.DataFrame, monthly_contrib: float = 0.0) -> pd.DataFrame:
    """
    $ delta to bring to target (synthetic MV), plus optional monthly contribution split.
    """
    df = df_alloc.copy()
    total = float(df["mkt_value"].sum())
    df["$to_target"] = (df["target_w"] - df["weight_now"]) * total

    if monthly_contrib > 0:
        df["$monthly_split"] = monthly_contrib * df["target_w"]
    else:
        df["$monthly_split"] = 0.0

    return df[[
        "symbol","yf_ticker","price","mkt_value","weight_now","target_w","band_lo","band_hi","status","$to_target","$monthly_split"
    ]]

# ============================ email ============================

def email_send(subject: str, body: str, attachments: list[tuple[str, pd.DataFrame]] | None = None) -> None:
    if EMAIL_DISABLE:
        _warn("EMAIL_DISABLE=1 -> skipping email")
        return
    if not all([SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS, MAIL_FROM, MAIL_TO]):
        _warn("Missing SMTP_* or MAIL_* env vars -> cannot email")
        return

    msg = EmailMessage()
    msg["From"] = MAIL_FROM
    msg["To"] = MAIL_TO
    msg["Subject"] = subject
    msg.set_content(body)

    for name, df in (attachments or []):
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        msg.add_attachment(buf.getvalue().encode("utf-8"), maintype="text", subtype="csv", filename=name)

    ctx = ssl.create_default_context()
    with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=30) as s:
        s.starttls(context=ctx)
        s.login(SMTP_USER, SMTP_PASS)
        s.send_message(msg)

def fmt_pct(x: float) -> str:
    return f"{x*100:6.2f}%"

def format_table(df: pd.DataFrame) -> str:
    lines = []
    lines.append(f"{'SYMBOL':<12} {'NOW':>8} {'TGT':>8} {'BAND':>18} {'STAT':>6}")
    for _, r in df.iterrows():
        band = f"[{r.band_lo*100:4.1f}–{r.band_hi*100:4.1f}%]"
        lines.append(f"{r.symbol:<12} {fmt_pct(r.weight_now):>8} {fmt_pct(r.target_w):>8} {band:>18} {r.status:>6}")
    return "\n".join(lines)

# ============================ jobs ============================

def daily_job():
    port = load_portfolio()
    yf_tickers = sorted(set([t for t in port["yf_ticker"].tolist() if t]))

    baseline = load_or_create_baseline(yf_tickers)

    data = _yf_download(yf_tickers, period="14d", interval="1d", tries=3)
    last_close = _last_row_series(data, "Close", yf_tickers)

    # as-of date from yfinance bar (if available)
    if isinstance(data, pd.DataFrame) and isinstance(data.index, pd.DatetimeIndex) and len(data.index) > 0:
        asof = pd.Timestamp(data.index[-1]).tz_localize(None).date().isoformat()
    else:
        asof = pd.Timestamp.utcnow().date().isoformat()

    alloc, total_mv = compute_allocation(port, last_close, baseline)

    # daily ROC from history MV
    hist = load_history()
    roc = 0.0
    if hist is not None and len(hist) >= 1:
        prev_mv = float(hist["mv"].iloc[-1])
        roc = (total_mv / prev_mv - 1.0) if prev_mv > 0 else 0.0

    upsert_history(asof, total_mv, roc)

    # lockup logic
    rebal_state = load_rebalance_state()
    in_lockup, days_left = lockup_status(asof, rebal_state)

    # signals
    breaches = alloc[alloc["would_trigger_now"]].copy()
    signals = {row["symbol"]: row["status"] for _, row in breaches.iterrows()}

    prev = _read_json(SIGNALS_JSON, {})
    if not isinstance(prev, dict):
        prev = {}

    changed = (signals != prev)
    _write_json(SIGNALS_JSON, signals)

    # status artifact
    status = {
        "as_of": asof,
        "baseline_date": baseline.get("as_of", ""),
        "notional_capital": NOTIONAL_CAPITAL,
        "total_value": total_mv,
        "daily_roc": roc,
        "last_rebalance_date": rebal_state.get("last_rebalance_date", ""),
        "locked_until": rebal_state.get("locked_until", ""),
        "lockup_days": int(rebal_state.get("lockup_days", LOCKUP_DAYS)),
        "in_lockup": bool(in_lockup),
        "lockup_days_remaining": int(days_left),
        "allocation": alloc.to_dict(orient="records"),
    }
    _write_json(STATUS_JSON, status)

    # email decision
    send = False
    reason = ""
    if breaches.empty:
        # If signals cleared, notify on change
        if prev and changed:
            send = True
            reason = "signals cleared"
    else:
        if ALWAYS_ALERT_ON_BREACH:
            send = True
            reason = "breach exists (always alert)"
        elif changed:
            send = True
            reason = "signal state changed"
        else:
            send = False

    # suppress during lockup (optional)
    if send and in_lockup:
        # during lockup: still update artifacts, but don't email (unless you want)
        send = False
        reason = f"suppressed by lockup ({days_left} days left)"

    if send:
        body = []
        body.append(f"As of: {asof}")
        body.append(f"Baseline: {baseline.get('as_of','')}")
        body.append(f"Total (synthetic): ${total_mv:,.2f} | Daily ROC: {roc*100:.2f}%")
        body.append("")
        body.append("Current allocation vs bands:")
        body.append(format_table(alloc))
        body.append("")
        if not breaches.empty:
            body.append("Breaches:")
            for _, r in breaches.iterrows():
                body.append(f"- {r.symbol}: {r.status} (now={r.weight_now*100:.2f}%, band {r.band_lo*100:.1f}–{r.band_hi*100:.1f}%)")
            body.append("")
        if rebal_state.get("locked_until"):
            body.append(f"Lockup: {LOCKUP_DAYS} calendar days | locked_until={rebal_state.get('locked_until')}")
        body.append(f"Reason: {reason}")

        reb = make_rebalance_deltas(alloc, monthly_contrib=2000.0)

        email_send(
            subject="Portfolio B — band signal update",
            body="\n".join(body),
            attachments=[
                ("allocation.csv", alloc[[
                    "symbol","yf_ticker","price","mkt_value","weight_now","target_w","band_lo","band_hi","status"
                ]]),
                ("rebalance_deltas.csv", reb),
            ],
        )
        print("[info] emailed:", reason, flush=True)
    else:
        print("[info] no email:", reason or "no state change / no breach", flush=True)

def weekly_job():
    port = load_portfolio()
    yf_tickers = sorted(set([t for t in port["yf_ticker"].tolist() if t]))
    baseline = load_or_create_baseline(yf_tickers)

    data = _yf_download(yf_tickers, period="14d", interval="1d", tries=3)
    last_close = _last_row_series(data, "Close", yf_tickers)

    if isinstance(data, pd.DataFrame) and isinstance(data.index, pd.DatetimeIndex) and len(data.index) > 0:
        asof = pd.Timestamp(data.index[-1]).tz_localize(None).date().isoformat()
    else:
        asof = pd.Timestamp.utcnow().date().isoformat()

    alloc, total_mv = compute_allocation(port, last_close, baseline)
    hist = load_history()

    def span_return(days: int) -> float:
        if hist is None or len(hist) < 2:
            return float("nan")
        end = float(hist["mv"].iloc[-1])
        start_idx = max(0, len(hist) - 1 - days)
        start = float(hist["mv"].iloc[start_idx])
        return (end / start - 1.0) if start > 0 else float("nan")

    ret_1w = span_return(5)
    ret_1m = span_return(21)

    rebal_state = load_rebalance_state()
    in_lockup, days_left = lockup_status(asof, rebal_state)

    body = []
    body.append(f"Weekly recap — as of {asof}")
    body.append(f"Baseline: {baseline.get('as_of','')}")
    body.append(f"Total (synthetic): ${total_mv:,.2f}")
    body.append("")
    body.append("Allocation:")
    body.append(format_table(alloc))
    body.append("")
    body.append("Returns (from history.csv synthetic MV):")
    body.append(f"- 1w: {ret_1w*100:,.2f}%" if np.isfinite(ret_1w) else "- 1w: n/a")
    body.append(f"- 1m: {ret_1m*100:,.2f}%" if np.isfinite(ret_1m) else "- 1m: n/a")
    body.append("")
    if rebal_state.get("last_rebalance_date"):
        body.append(f"Last recorded rebalance: {rebal_state.get('last_rebalance_date')}")
        body.append(f"Locked until: {rebal_state.get('locked_until')} ({days_left}d remaining)" if in_lockup else f"Locked until: {rebal_state.get('locked_until')}")
    else:
        body.append("Last recorded rebalance: (none recorded)")

    reb = make_rebalance_deltas(alloc, monthly_contrib=2000.0)

    email_send(
        subject="Portfolio B — weekly recap",
        body="\n".join(body),
        attachments=[
            ("allocation.csv", alloc[[
                "symbol","yf_ticker","price","mkt_value","weight_now","target_w","band_lo","band_hi","status"
            ]]),
            ("rebalance_deltas.csv", reb),
            ("history.csv", hist if hist is not None else pd.DataFrame()),
        ],
    )
    print("[info] weekly email sent", flush=True)

def mark_rebalanced():
    """
    Record that you actually rebalanced (manually) as-of the latest date in status.json if present,
    otherwise today's UTC date.
    """
    st = load_rebalance_state()
    status = _read_json(STATUS_JSON, {})
    asof = status.get("as_of", "") if isinstance(status, dict) else ""
    if not asof:
        asof = pd.Timestamp.utcnow().date().isoformat()

    last = pd.to_datetime(asof)
    locked_until = (last + pd.Timedelta(days=LOCKUP_DAYS)).date().isoformat()

    st["last_rebalance_date"] = asof
    st["lockup_days"] = LOCKUP_DAYS
    st["locked_until"] = locked_until
    save_rebalance_state(st)

    print(f"[info] marked rebalanced on {asof}; lockup until {locked_until} ({LOCKUP_DAYS} calendar days)")

def main():
    mode = (sys.argv[1] if len(sys.argv) > 1 else "daily").lower()
    try:
        if mode == "daily":
            daily_job()
        elif mode == "weekly":
            weekly_job()
        elif mode in ("mark_rebalanced", "rebalance", "mark"):
            mark_rebalanced()
        else:
            print("Usage: python tracker.py [daily|weekly|mark_rebalanced]")
            sys.exit(2)
    except Exception as e:
        _err("crashed:", repr(e))
        raise

if __name__ == "__main__":
    main()
