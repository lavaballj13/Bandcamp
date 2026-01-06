#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BAND + LOCKUP Portfolio Tracker (FULL DROP-IN)

What it does (GitHub Actions friendly):
- Reads portfolio.csv (targets + per-asset band_down/band_up like your optimizer).
- Builds a baseline.json (first run) with baseline Close + Adj Close.
- Computes current synthetic drift weights vs targets (weights-only; no shares required).
- Detects band breaches.
- Enforces a CALENDAR-DAY lockup (same semantics as your optimizer):
    locked if (as_of_date - last_rebalance_date).days < LOCKUP_DAYS
- Email includes:
    - last_rebalance_date
    - locked_until
    - days_left
    - breach list
    - whether breaches are suppressed by lockup
    - whether an auto-rebalance executed today
- Optional AUTO “virtual rebalance” when breach occurs AND not locked:
    - updates last_rebalance_date in rebalance_state.json
    - resets baseline prices to today’s prices (so weights snap back to target)
    - starts new lockup window

Artifacts committed by workflow:
- baseline.json
- rebalance_state.json
- signals_state.json
- status.json
- history.csv

Required files:
- portfolio.csv

Install deps:
- pandas numpy yfinance

Environment (optional):
- NOTIONAL_CAPITAL=1000
- BASELINE_DATE=YYYY-MM-DD or "" (blank => first run uses latest bar)
- LOCKUP_DAYS=365
- AUTO_REBALANCE_ON_BREACH=0/1
- ALWAYS_ALERT_ON_BREACH=0/1
- TRACKER_DEBUG=0/1
- CONTRIBUTION_AMOUNT=2000  (used in weekly recap buy-only plan)
- EMAIL_DISABLE=0/1  (skip email)
- SMTP_HOST/SMTP_PORT/SMTP_USER/SMTP_PASS/MAIL_FROM/MAIL_TO
"""

import os, io, json, smtplib, ssl, sys, math, time
from email.message import EmailMessage
from pathlib import Path

import pandas as pd
import numpy as np
import yfinance as yf

# =======================
# Config
# =======================
NOTIONAL_CAPITAL = float(os.getenv("NOTIONAL_CAPITAL", "1000.0"))
BASELINE_DATE    = os.getenv("BASELINE_DATE", "")  # "" => set on first run (latest bar)
LOCKUP_DAYS      = int(os.getenv("LOCKUP_DAYS", "365"))  # CALENDAR DAYS (matches your optimizer)
AUTO_REBALANCE_ON_BREACH = (os.getenv("AUTO_REBALANCE_ON_BREACH", "0") == "1")
ALWAYS_ALERT_ON_BREACH   = (os.getenv("ALWAYS_ALERT_ON_BREACH", "0") == "1")
CONTRIBUTION_AMOUNT      = float(os.getenv("CONTRIBUTION_AMOUNT", "2000"))

DBG = (os.getenv("TRACKER_DEBUG", "0") == "1")
EPS = 1e-9

# =======================
# Files
# =======================
PORTFOLIO_CSV = "portfolio.csv"
BASELINE_JSON = "baseline.json"
HISTORY_CSV   = "history.csv"
STATUS_JSON   = "status.json"
SIGNALS_JSON  = "signals_state.json"
REBALANCE_STATE_JSON = "rebalance_state.json"

# =======================
# Email (GitHub Secrets)
# =======================
SMTP_HOST = os.getenv("SMTP_HOST")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASS = os.getenv("SMTP_PASS")
MAIL_FROM = os.getenv("MAIL_FROM")
MAIL_TO   = os.getenv("MAIL_TO")
EMAIL_DISABLE = (os.getenv("EMAIL_DISABLE", "0") == "1")

# =======================
# Logging
# =======================
def _dbg(*a):
    if DBG:
        print("[dbg]", *a, flush=True)

def _warn(*a):
    print("[warn]", *a, flush=True)

def _err(*a):
    print("[error]", *a, flush=True)

# =======================
# Portfolio I/O
# =======================
def load_portfolio(path=PORTFOLIO_CSV) -> pd.DataFrame:
    """
    portfolio.csv columns:
      ticker,target_w,band_down,band_up

    band_down/band_up are ABSOLUTE deviations (e.g. 0.10 = 10%) like your optimizer.
    Use blank/0 for no band (i.e., never triggers).
    """
    df = pd.read_csv(path)

    need = {"ticker", "target_w", "band_down", "band_up"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")

    df = df.copy()
    df["ticker"] = df["ticker"].astype(str).str.strip()

    for c in ["target_w", "band_down", "band_up"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    # normalize targets to sum 1
    tw = df["target_w"].to_numpy(float)
    s = float(np.nansum(tw))
    if s <= 0:
        raise ValueError("portfolio.csv target_w sum must be > 0")
    if not np.isclose(s, 1.0):
        df["target_w"] = tw / s

    # basic sanity
    if (df["target_w"] < -EPS).any():
        raise ValueError("portfolio.csv has negative target_w")
    if (df["band_down"] < -EPS).any() or (df["band_up"] < -EPS).any():
        raise ValueError("portfolio.csv has negative band_down/band_up")

    _dbg("portfolio.csv:\n", df.to_string(index=False))
    return df

# =======================
# yfinance helpers
# =======================
def _yf_download(tickers, period="10d", interval="1d", tries=3) -> pd.DataFrame:
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

    _warn(f"yfinance download empty after {tries} tries (period={period}, interval={interval}) err={last_err}")
    return pd.DataFrame()

def _get_field_safe(frame: pd.DataFrame, field: str) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or len(frame) == 0:
        raise KeyError("empty frame")

    if isinstance(frame.columns, pd.MultiIndex):
        # Multi-ticker
        if field in frame.columns.get_level_values(0):
            return frame[field]
        if field == "Adj Close" and "Close" in frame.columns.get_level_values(0):
            _warn("Adj Close missing; falling back to Close")
            return frame["Close"]
        raise KeyError(f"{field!r} missing in MultiIndex columns")

    # Single ticker
    if field in frame.columns:
        return frame[[field]]
    if field == "Adj Close" and "Close" in frame.columns:
        _warn("Adj Close missing; falling back to Close")
        return frame[["Close"]]
    raise KeyError(f"{field!r} missing in columns {frame.columns.tolist()}")

def _last_series(frame: pd.DataFrame, field: str, tickers: list[str]) -> pd.Series:
    try:
        sub = _get_field_safe(frame, field)
    except Exception:
        return pd.Series({t: np.nan for t in tickers}, dtype="float64")

    if isinstance(frame.columns, pd.MultiIndex):
        ser = sub.iloc[-1]
        ser.index = ser.index.astype(str)
        return pd.Series(ser, dtype="float64").reindex(tickers)
    else:
        # single ticker
        val = float(sub.iloc[-1, 0])
        return pd.Series({tickers[0]: val}, dtype="float64")

def _two_series(frame: pd.DataFrame, field: str, tickers: list[str]) -> tuple[pd.Series, pd.Series]:
    if not isinstance(frame, pd.DataFrame) or len(frame) == 0:
        s = pd.Series({t: np.nan for t in tickers}, dtype="float64")
        return s, s
    if len(frame) == 1:
        s = _last_series(frame, field, tickers)
        return s, s

    try:
        sub = _get_field_safe(frame, field)
    except Exception:
        s = pd.Series({t: np.nan for t in tickers}, dtype="float64")
        return s, s

    if isinstance(frame.columns, pd.MultiIndex):
        prev_ = sub.iloc[-2]; now_ = sub.iloc[-1]
        prev_.index = prev_.index.astype(str); now_.index = now_.index.astype(str)
        return (pd.Series(prev_, dtype="float64").reindex(tickers),
                pd.Series(now_,  dtype="float64").reindex(tickers))
    else:
        return (pd.Series({tickers[0]: float(sub.iloc[-2, 0])}, dtype="float64"),
                pd.Series({tickers[0]: float(sub.iloc[-1, 0])}, dtype="float64"))

# =======================
# Baseline handling
# =======================
def load_or_create_baseline(tickers: list[str], baseline_date: str = BASELINE_DATE) -> dict:
    if Path(BASELINE_JSON).exists():
        try:
            js = json.loads(Path(BASELINE_JSON).read_text())
            # minimal validation
            if "base_close" in js and "base_adj" in js:
                return js
        except Exception as e:
            _warn(f"baseline.json unreadable; recreating. err={e}")

    # Create baseline
    if baseline_date:
        # Try a window starting at baseline_date (yfinance may return from next trading day)
        hist = yf.download(
            tickers=tickers,
            start=baseline_date,
            interval="1d",
            auto_adjust=False,
            progress=False,
            group_by="column",
        )
        if not isinstance(hist, pd.DataFrame) or len(hist) == 0:
            # fallback: recent period then pick first >= baseline_date if possible
            hist = _yf_download(tickers, period="90d", interval="1d", tries=3)
            if not isinstance(hist, pd.DataFrame) or len(hist) == 0:
                raise RuntimeError("Could not fetch data to create baseline.")
            try:
                base_dt_req = pd.to_datetime(baseline_date)
                hist2 = hist[hist.index >= base_dt_req]
                if len(hist2) > 0:
                    hist = hist2
            except Exception:
                pass

        base_dt = pd.Timestamp(hist.index[0]).tz_localize(None)
        close0 = _get_field_safe(hist, "Close").iloc[0]
        try:
            adj0 = _get_field_safe(hist, "Adj Close").iloc[0]
        except Exception:
            adj0 = close0

    else:
        # baseline = latest bar
        hist = _yf_download(tickers, period="10d", interval="1d", tries=3)
        if not isinstance(hist, pd.DataFrame) or len(hist) == 0:
            raise RuntimeError("Could not fetch data to create baseline.")
        base_dt = pd.Timestamp(hist.index[-1]).tz_localize(None)
        close0 = _get_field_safe(hist, "Close").iloc[-1]
        try:
            adj0 = _get_field_safe(hist, "Adj Close").iloc[-1]
        except Exception:
            adj0 = close0

    if isinstance(hist.columns, pd.MultiIndex):
        close0.index = close0.index.astype(str)
        adj0.index = adj0.index.astype(str)
        base_close = {t: float(close0.get(t, np.nan)) for t in tickers}
        base_adj   = {t: float(adj0.get(t, np.nan))   for t in tickers}
    else:
        base_close = {tickers[0]: float(close0)}
        base_adj   = {tickers[0]: float(adj0)}

    baseline = {
        "as_of": base_dt.date().isoformat(),
        "capital": float(NOTIONAL_CAPITAL),
        "base_close": base_close,
        "base_adj": base_adj,
    }
    Path(BASELINE_JSON).write_text(json.dumps(baseline, indent=2))
    return baseline

# =======================
# Lockup / rebalance state
# =======================
def load_json(path: str) -> dict:
    p = Path(path)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return {}
    return {}

def save_json(path: str, obj: dict) -> None:
    Path(path).write_text(json.dumps(obj, indent=2))

def lockup_info(as_of_dt: pd.Timestamp, rebal_state: dict) -> dict:
    """
    lockup is CALENDAR DAYS: locked if days_since_last < LOCKUP_DAYS
    """
    last = rebal_state.get("last_rebalance_date")  # YYYY-MM-DD
    if not last:
        return {
            "last_rebalance_date": None,
            "locked_until": None,
            "is_locked": False,
            "days_left": 0,
        }

    last_dt = pd.to_datetime(last).normalize()
    locked_until_dt = (last_dt + pd.Timedelta(days=int(LOCKUP_DAYS))).normalize()
    asof = pd.to_datetime(as_of_dt).normalize()

    is_locked = asof < locked_until_dt
    days_left = int((locked_until_dt - asof).days) if is_locked else 0

    return {
        "last_rebalance_date": last_dt.date().isoformat(),
        "locked_until": locked_until_dt.date().isoformat(),
        "is_locked": bool(is_locked),
        "days_left": int(days_left),
    }

def perform_virtual_rebalance(
    tickers: list[str],
    as_of_dt: pd.Timestamp,
    last_close: pd.Series,
    last_adj: pd.Series,
) -> None:
    """
    Virtual rebalance for a weights-only tracker:
    - Update baseline.json base_close/base_adj to today's values
      so drift snaps back to target weights.
    - Update rebalance_state.json last_rebalance_date
    """
    baseline = load_json(BASELINE_JSON) if Path(BASELINE_JSON).exists() else {}
    baseline["as_of"] = as_of_dt.date().isoformat()
    baseline["capital"] = float(baseline.get("capital", NOTIONAL_CAPITAL))

    baseline["base_close"] = {t: float(last_close.reindex([t]).iloc[0]) for t in tickers}
    baseline["base_adj"]   = {t: float(last_adj.reindex([t]).iloc[0])   for t in tickers}

    save_json(BASELINE_JSON, baseline)

    st = load_json(REBALANCE_STATE_JSON)
    st["last_rebalance_date"] = as_of_dt.date().isoformat()
    save_json(REBALANCE_STATE_JSON, st)

# =======================
# Core math (bands like optimizer)
# =======================
def compute_alloc(port: pd.DataFrame, last_close: pd.Series, baseline: dict) -> tuple[pd.DataFrame, float]:
    tickers = port["ticker"].tolist()

    base_close = pd.Series({t: baseline["base_close"].get(t, np.nan) for t in tickers}, dtype="float64")
    last_vec   = last_close.reindex(tickers).astype("float64")

    idx = (last_vec / base_close).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)

    capital = float(baseline.get("capital", NOTIONAL_CAPITAL))
    targ = port["target_w"].to_numpy(float)

    mv = capital * targ * idx.to_numpy(float)
    total_mv = float(np.nansum(mv))
    w_now = (mv / total_mv) if total_mv > 0 else np.zeros_like(mv)

    out = port.copy()
    out["price"] = last_vec.to_numpy(float)
    out["mkt_value"] = mv
    out["weight_now"] = w_now

    # compute band bounds as target +/- (down/up) (clamped)
    out["band_lo"] = np.clip(out["target_w"] - out["band_down"], 0.0, 1.0)
    out["band_hi"] = np.clip(out["target_w"] + out["band_up"],   0.0, 1.0)

    def status_row(r):
        # no band => OK (never triggers)
        if float(r["band_down"]) <= EPS and float(r["band_up"]) <= EPS:
            return "OK"
        if float(r["weight_now"]) < float(r["band_lo"]) - EPS:
            return "LOW"
        if float(r["weight_now"]) > float(r["band_hi"]) + EPS:
            return "HIGH"
        return "OK"

    out["status"] = out.apply(status_row, axis=1)

    out["delta_w"] = out["weight_now"] - out["target_w"]
    out["would_trigger_now"] = out["status"].isin(["LOW", "HIGH"])

    _dbg("alloc:\n", out.to_string(index=False))
    return out, total_mv

def detect_breaches(df_alloc: pd.DataFrame) -> list[dict]:
    breaches = []
    for _, r in df_alloc.iterrows():
        if bool(r["would_trigger_now"]):
            breaches.append({
                "ticker": str(r["ticker"]),
                "status": str(r["status"]),
                "weight_now": float(r["weight_now"]),
                "target_w": float(r["target_w"]),
                "band_lo": float(r["band_lo"]),
                "band_hi": float(r["band_hi"]),
            })
    return breaches

def make_rebalance_table(df_alloc: pd.DataFrame, to: str = "target") -> pd.DataFrame:
    total_mv = float(df_alloc["mkt_value"].sum())
    w_now = df_alloc["weight_now"].to_numpy(float)

    if to == "band_mid":
        w_t = 0.5 * (df_alloc["band_lo"].to_numpy(float) + df_alloc["band_hi"].to_numpy(float))
        # if no band, mid==target
        no_band = (df_alloc["band_down"].to_numpy(float) <= EPS) & (df_alloc["band_up"].to_numpy(float) <= EPS)
        w_t[no_band] = df_alloc["target_w"].to_numpy(float)[no_band]
    else:
        w_t = df_alloc["target_w"].to_numpy(float)

    delta_dollars = (w_t - w_now) * total_mv
    out = df_alloc[["ticker", "price", "mkt_value", "weight_now", "target_w", "band_lo", "band_hi", "status"]].copy()
    out["rebalance_to_w"] = w_t
    out["$delta"] = delta_dollars
    return out.sort_values("$delta")

def buy_only_contribution_plan(df_alloc: pd.DataFrame, amount: float) -> pd.DataFrame:
    """
    Buy-only plan: allocate contribution to underweight assets (target - now > 0)
    proportional to deficit size.
    """
    if amount <= 0:
        return pd.DataFrame(columns=["ticker", "buy_$", "buy_w_of_contribution", "deficit_w"])

    deficit = (df_alloc["target_w"] - df_alloc["weight_now"]).clip(lower=0.0)
    tot = float(deficit.sum())
    if tot <= EPS:
        # If nothing is underweight, just allocate by target weights
        alloc_w = df_alloc["target_w"].to_numpy(float)
    else:
        alloc_w = (deficit / tot).to_numpy(float)

    buy_dollars = amount * alloc_w
    out = pd.DataFrame({
        "ticker": df_alloc["ticker"].astype(str),
        "buy_$": buy_dollars,
        "buy_w_of_contribution": alloc_w,
        "deficit_w": deficit.to_numpy(float),
    })
    return out.sort_values("buy_$", ascending=False)

# =======================
# History / snapshots
# =======================
def _collapse_history_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return df
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")
    g = df.groupby(df["date"].dt.date, as_index=False).last()
    g["date"] = pd.to_datetime(g["date"])
    return g

def update_history(dt_label: str, mv_price: float, roc_price: float, mv_tr: float, roc_tr: float) -> None:
    cols = ["date", "mv_price", "roc_price", "mv_tr", "roc_total_return"]
    day = pd.to_datetime(str(dt_label)).date().isoformat()
    new_row = pd.DataFrame([{
        "date": day,
        "mv_price": float(mv_price),
        "roc_price": float(roc_price),
        "mv_tr": float(mv_tr),
        "roc_total_return": float(roc_tr),
    }], columns=cols)

    p = Path(HISTORY_CSV)
    if p.exists():
        try:
            df = pd.read_csv(p, parse_dates=["date"])
        except Exception:
            df = pd.DataFrame(columns=cols)
        df = _collapse_history_df(df)
        if len(df):
            df = df[df["date"].dt.date.astype(str) != day]
        df = pd.concat([df, new_row], ignore_index=True)
    else:
        df = new_row

    df = df.sort_values("date")
    df_out = df.copy()
    df_out["date"] = pd.to_datetime(df_out["date"]).dt.strftime("%Y-%m-%d")
    df_out.to_csv(p, index=False)

# =======================
# Email
# =======================
def format_email_table(df: pd.DataFrame, title: str) -> str:
    lines = [title, "-" * len(title)]
    lines.append(f"{'Ticker':<10}{'Now':>8}  {'Target':>8}  {'Band':>16}  {'Status':>7}")
    for _, r in df.iterrows():
        band = f"[{r.band_lo*100:5.1f}–{r.band_hi*100:5.1f}%]" if (r.band_down > EPS or r.band_up > EPS) else "None"
        lines.append(
            f"{str(r.ticker):<10}{r.weight_now*100:>7.2f}%  "
            f"{r.target_w*100:>7.1f}%  "
            f"{band:>16}  {str(r.status):>7}"
        )
    return "\n".join(lines)

def email_send(subject: str, body: str, attachments: list[tuple[str, pd.DataFrame]] | None = None) -> None:
    if EMAIL_DISABLE:
        _warn("EMAIL_DISABLE=1 -> skipping email send")
        return
    if not all([SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS, MAIL_FROM, MAIL_TO]):
        _warn("Email not configured; set SMTP_* and MAIL_* secrets.")
        return

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
            filename=name
        )

    ctx = ssl.create_default_context()
    with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=30) as s:
        s.starttls(context=ctx)
        s.login(SMTP_USER, SMTP_PASS)
        s.send_message(msg)

    print("[info] email sent:", subject, flush=True)

# =======================
# Signals state (for “changed” detection)
# =======================
def load_signals_state() -> dict:
    return load_json(SIGNALS_JSON)

def save_signals_state(state: dict) -> None:
    save_json(SIGNALS_JSON, state)

# =======================
# Jobs
# =======================
def daily_job() -> None:
    port = load_portfolio()
    tickers = port["ticker"].tolist()

    baseline = load_or_create_baseline(tickers, baseline_date=BASELINE_DATE)

    data = _yf_download(tickers, period="10d", interval="1d", tries=3)
    last_close = _last_series(data, "Close", tickers)
    last_adj   = _last_series(data, "Adj Close", tickers)

    if isinstance(data.index, pd.DatetimeIndex) and len(data.index) >= 1:
        last_dt = pd.Timestamp(data.index[-1]).tz_localize(None)
    else:
        last_dt = pd.Timestamp.utcnow().tz_localize(None)

    # compute current alloc
    alloc, mv_now_price = compute_alloc(port, last_close, baseline)
    breaches = detect_breaches(alloc)

    # compute daily ROC (price & TR proxy)
    d2 = _yf_download(tickers, period="4d", interval="1d", tries=3)
    c_prev, c_now = _two_series(d2, "Close", tickers)
    a_prev, a_now = _two_series(d2, "Adj Close", tickers)

    base_close_ser = pd.Series(baseline["base_close"], dtype="float64").reindex(tickers)
    base_adj_ser   = pd.Series(baseline["base_adj"],   dtype="float64").reindex(tickers)
    targ           = port["target_w"].to_numpy(float)
    capital        = float(baseline.get("capital", NOTIONAL_CAPITAL))

    mv_prev_price = float(capital * np.nansum(targ * (c_prev.to_numpy() / base_close_ser.to_numpy())))
    mv_now_price2 = float(capital * np.nansum(targ * (c_now.to_numpy()  / base_close_ser.to_numpy())))
    roc_price = (mv_now_price2 / mv_prev_price - 1.0) if mv_prev_price > 0 else 0.0

    mv_prev_tr = float(capital * np.nansum(targ * (a_prev.to_numpy() / base_adj_ser.to_numpy())))
    mv_now_tr  = float(capital * np.nansum(targ * (a_now.to_numpy()  / base_adj_ser.to_numpy())))
    roc_tr = (mv_now_tr / mv_prev_tr - 1.0) if mv_prev_tr > 0 else 0.0

    # lockup
    rebal_state = load_json(REBALANCE_STATE_JSON)
    lk = lockup_info(last_dt, rebal_state)

    breach_exists = bool(breaches)
    suppressed_by_lockup = breach_exists and lk["is_locked"]

    did_rebalance = False
    if breach_exists and (not lk["is_locked"]) and AUTO_REBALANCE_ON_BREACH:
        # execute virtual rebalance, then recompute alloc/breaches and lockup
        perform_virtual_rebalance(tickers, last_dt, last_close, last_adj)
        did_rebalance = True

        baseline = load_json(BASELINE_JSON)
        alloc, mv_now_price = compute_alloc(port, last_close, baseline)
        breaches = detect_breaches(alloc)

        rebal_state = load_json(REBALANCE_STATE_JSON)
        lk = lockup_info(last_dt, rebal_state)

    # state-change logic (only email on change unless ALWAYS_ALERT_ON_BREACH=1)
    prev_state = load_signals_state()
    today_state = {b["ticker"]: b["status"] for b in breaches}  # only breached tickers are included
    changed = (today_state != prev_state)
    save_signals_state(today_state)

    # history + status snapshot
    update_history(last_dt.date().isoformat(), mv_now_price2, roc_price, mv_now_tr, roc_tr)

    snapshot = {
        "as_of": last_dt.date().isoformat(),
        "baseline_date": baseline.get("as_of"),
        "capital": float(baseline.get("capital", NOTIONAL_CAPITAL)),
        "total_mv_price": float(mv_now_price2),
        "roc_price": float(roc_price),
        "roc_total_return": float(roc_tr),

        "lockup_days": int(LOCKUP_DAYS),
        "last_rebalance_date": lk["last_rebalance_date"],
        "locked_until": lk["locked_until"],
        "is_locked": bool(lk["is_locked"]),
        "lockup_days_left": int(lk["days_left"]),
        "did_rebalance_today": bool(did_rebalance),
        "breach_suppressed_by_lockup": bool(suppressed_by_lockup),

        "breaches": breaches,
        "allocation": alloc[[
            "ticker","price","mkt_value","weight_now","target_w",
            "band_down","band_up","band_lo","band_hi","delta_w","status","would_trigger_now"
        ]].to_dict(orient="records"),
    }
    save_json(STATUS_JSON, snapshot)

    # Email rules:
    # - If breaches exist: email when state changes OR ALWAYS_ALERT_ON_BREACH
    # - If auto-rebalanced: email (so you see it happened), even if breaches cleared
    should_email = False
    if breach_exists and (ALWAYS_ALERT_ON_BREACH or changed):
        should_email = True
    if did_rebalance:
        should_email = True

    if should_email:
        body = ""
        body += f"As of {last_dt.date().isoformat()} (EOD)\n"
        body += f"Baseline date: {baseline.get('as_of')}\n\n"

        body += f"Lockup (calendar): {LOCKUP_DAYS} days\n"
        body += f"Last rebalance:     {lk['last_rebalance_date'] or 'N/A'}\n"
        body += f"Locked until:       {lk['locked_until'] or 'N/A'}"
        if lk["is_locked"]:
            body += f"  (days left: {lk['days_left']})\n"
        else:
            body += "\n"
        body += f"Auto rebalance:     {'YES' if AUTO_REBALANCE_ON_BREACH else 'NO'}\n"
        body += f"Rebalanced today:   {'YES' if did_rebalance else 'NO'}\n"
        if suppressed_by_lockup:
            body += "⛔ Breach detected but suppressed due to lockup.\n"
        body += "\n"

        body += format_email_table(alloc, "Current Allocation") + "\n\n"

        if breaches:
            body += "⚠️ Band breaches:\n"
            for b in breaches:
                body += (
                    f"- {b['ticker']}: {b['status']} "
                    f"(w={b['weight_now']*100:.2f}% vs "
                    f"band {b['band_lo']*100:.1f}–{b['band_hi']*100:.1f}%, "
                    f"target {b['target_w']*100:.1f}%)\n"
                )
            body += "\n"
        else:
            body += "No band breaches.\n\n"

        reb_to_target = make_rebalance_table(alloc, to="target")
        reb_to_mid    = make_rebalance_table(alloc, to="band_mid")

        subj = "Portfolio — SIGNALS" if breaches else ("Portfolio — Rebalanced" if did_rebalance else "Portfolio — Daily")
        email_send(
            subject=subj,
            body=body,
            attachments=[
                ("allocation.csv", alloc),
                ("rebalance_to_target.csv", reb_to_target),
                ("rebalance_to_band_mid.csv", reb_to_mid),
            ],
        )
    else:
        print("[info] no email sent (no breaches, and no state change).", flush=True)

def weekly_job() -> None:
    port = load_portfolio()
    tickers = port["ticker"].tolist()

    baseline = load_or_create_baseline(tickers, baseline_date=BASELINE_DATE)

    data = _yf_download(tickers, period="10d", interval="1d", tries=3)
    last_close = _last_series(data, "Close", tickers)
    last_adj   = _last_series(data, "Adj Close", tickers)

    if isinstance(data.index, pd.DatetimeIndex) and len(data.index) >= 1:
        last_dt = pd.Timestamp(data.index[-1]).tz_localize(None)
    else:
        last_dt = pd.Timestamp.utcnow().tz_localize(None)

    alloc, _ = compute_alloc(port, last_close, baseline)

    # lockup
    rebal_state = load_json(REBALANCE_STATE_JSON)
    lk = lockup_info(last_dt, rebal_state)

    # history-derived perf
    hist = None
    if Path(HISTORY_CSV).exists():
        try:
            hist = pd.read_csv(HISTORY_CSV, parse_dates=["date"])
            hist = _collapse_history_df(hist)
        except Exception as e:
            _warn(f"failed reading history.csv: {e}")

    def span_return(colname: str, days: int) -> float:
        if hist is None or len(hist) < 2:
            return float("nan")
        end = float(hist[colname].iloc[-1])
        start_idx = max(0, len(hist) - 1 - days)
        start = float(hist[colname].iloc[start_idx])
        return (end / start - 1.0) if start > 0 else float("nan")

    ret_1w = span_return("mv_tr", 5)

    ret_mtd = float("nan")
    ret_ytd = float("nan")
    if hist is not None and len(hist) >= 2:
        # MTD
        this_month = hist[hist["date"].dt.to_period("M") == hist["date"].iloc[-1].to_period("M")]
        if len(this_month) >= 2:
            ret_mtd = float(this_month["mv_tr"].iloc[-1] / this_month["mv_tr"].iloc[0] - 1.0)
        # YTD
        this_year = hist[hist["date"].dt.year == hist["date"].iloc[-1].year]
        if len(this_year) >= 2:
            ret_ytd = float(this_year["mv_tr"].iloc[-1] / this_year["mv_tr"].iloc[0] - 1.0)

    breaches = detect_breaches(alloc)

    body = ""
    body += f"Weekly recap — as of {last_dt.date().isoformat()}\n"
    body += f"Baseline date: {baseline.get('as_of')}\n\n"

    body += f"Lockup (calendar): {LOCKUP_DAYS} days\n"
    body += f"Last rebalance:     {lk['last_rebalance_date'] or 'N/A'}\n"
    body += f"Locked until:       {lk['locked_until'] or 'N/A'}"
    if lk["is_locked"]:
        body += f"  (days left: {lk['days_left']})\n\n"
    else:
        body += "\n\n"

    body += format_email_table(alloc, "Current Allocation") + "\n\n"

    body += "Performance (Adj Close proxy TR):\n"
    body += (f"- 1w:  {ret_1w*100:6.2f}%\n" if np.isfinite(ret_1w) else "- 1w:  n/a\n")
    body += (f"- MTD: {ret_mtd*100:6.2f}%\n" if np.isfinite(ret_mtd) else "- MTD: n/a\n")
    body += (f"- YTD: {ret_ytd*100:6.2f}%\n" if np.isfinite(ret_ytd) else "- YTD: n/a\n")
    body += "\n"

    if breaches:
        body += "⚠️ Current breaches:\n"
        for b in breaches:
            body += (
                f"- {b['ticker']}: {b['status']} "
                f"(w={b['weight_now']*100:.2f}% vs "
                f"band {b['band_lo']*100:.1f}–{b['band_hi']*100:.1f}%, "
                f"target {b['target_w']*100:.1f}%)\n"
            )
        body += "\n"

    reb_to_target = make_rebalance_table(alloc, to="target")
    reb_to_mid    = make_rebalance_table(alloc, to="band_mid")
    buy_plan      = buy_only_contribution_plan(alloc, CONTRIBUTION_AMOUNT)

    body += f"Buy-only plan for contribution: ${CONTRIBUTION_AMOUNT:,.2f}\n"
    body += "(allocates only to underweights; if none underweight, allocates by targets)\n"

    email_send(
        subject="Portfolio — Weekly recap",
        body=body,
        attachments=[
            ("allocation.csv", alloc),
            ("rebalance_to_target.csv", reb_to_target),
            ("rebalance_to_band_mid.csv", reb_to_mid),
            ("buy_only_contribution_plan.csv", buy_plan),
            ("history.csv", pd.read_csv(HISTORY_CSV) if Path(HISTORY_CSV).exists() else pd.DataFrame()),
        ],
    )

# =======================
# CLI
# =======================
def main():
    mode = (sys.argv[1] if len(sys.argv) > 1 else "daily").lower()
    if mode == "daily":
        daily_job()
    elif mode == "weekly":
        weekly_job()
    else:
        print("Usage: python tracker.py [daily|weekly]")
        sys.exit(2)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        _err("tracker crashed:", repr(e))
        raise
