#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Weights-only portfolio tracker
- No real shares required. We track drift vs targets using baseline prices.
- First run creates baseline.json with baseline Close & Adj Close per ticker.
- Daily EOD: compute current weights, detect band breaches ("signals"), email if signals appear/clear.
- Weekly: recap email with performance and suggested $ deltas to reach targets.

Artifacts:
- baseline.json        (created on first run if missing)
- history.csv          (daily MV logs; price & TR proxy)
- status.json          (latest snapshot)
- signals_state.json   (to detect "state changed" signals)
"""

import os, io, json, smtplib, ssl, sys
from email.message import EmailMessage
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import numpy as np
import yfinance as yf

# ---------- User-configurable knobs ----------
NOTIONAL_CAPITAL   = 1000.0                 # $ used for synthetic portfolio value calc
BASELINE_DATE = "2025-07-17"
TZ_USER            = ZoneInfo("America/Denver")  # your local TZ for display
ALWAYS_ALERT_ON_BREACH = False              # True => email any day breaches exist; False => only when state changes

# ---------- Files ----------
PORTFOLIO_CSV = "portfolio.csv"
BASELINE_JSON = "baseline.json"
HISTORY_CSV   = "history.csv"
STATUS_JSON   = "status.json"
SIGNALS_JSON  = "signals_state.json"

# ---------- Email (from GitHub Secrets) ----------
SMTP_HOST = os.environ.get("SMTP_HOST")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
SMTP_USER = os.environ.get("SMTP_USER")
SMTP_PASS = os.environ.get("SMTP_PASS")
MAIL_FROM = os.environ.get("MAIL_FROM")
MAIL_TO   = os.environ.get("MAIL_TO")

EPS = 1e-9

# ============================ Core helpers ============================

def load_portfolio(path=PORTFOLIO_CSV) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = {"ticker","target_w","band_lo","band_hi"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    df["ticker"] = df["ticker"].astype(str)
    for c in ["target_w","band_lo","band_hi"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # normalize targets if needed
    tw = df["target_w"].to_numpy(float)
    if not np.isclose(np.nansum(tw), 1.0):
        tw = tw / np.nansum(tw)
        df["target_w"] = tw
    return df

def _yf_download(tickers, period="7d", interval="1d"):
    return yf.download(
        tickers=tickers, period=period, interval=interval,
        auto_adjust=False, progress=False, group_by="column"
    )

def _last_series(frame, field, tickers):
    """Get last row Series for field ('Close' / 'Adj Close') from yfinance frame."""
    if isinstance(frame.columns, pd.MultiIndex):
        ser = frame[field].iloc[-1]
        ser.index = ser.index.astype(str)
        return pd.Series(ser, dtype="float64").reindex(tickers)
    else:
        return pd.Series({tickers[0]: frame[field].iloc[-1]})

def _two_series(frame, field, tickers):
    """Get last two rows Series for field."""
    if isinstance(frame.columns, pd.MultiIndex):
        prev_ = frame[field].iloc[-2]
        now_  = frame[field].iloc[-1]
        prev_.index = prev_.index.astype(str)
        now_.index  = now_.index.astype(str)
        return (pd.Series(prev_, dtype="float64").reindex(tickers),
                pd.Series(now_,  dtype="float64").reindex(tickers))
    else:
        return (pd.Series({tickers[0]: frame[field].iloc[-2]}),
                pd.Series({tickers[0]: frame[field].iloc[-1]}))

def load_or_create_baseline(tickers, baseline_date=BASELINE_DATE):
    """Load baseline if exists; else create using baseline_date (or last bar if blank)."""
    if Path(BASELINE_JSON).exists():
        return json.loads(Path(BASELINE_JSON).read_text())

    # Need to create baseline
    if baseline_date:
        hist = yf.download(tickers=tickers, start=baseline_date, end=None,
                           interval="1d", auto_adjust=False, progress=False, group_by="column")
        if not isinstance(hist.index, pd.DatetimeIndex) or not len(hist.index):
            raise RuntimeError("Could not fetch data to create baseline; check BASELINE_DATE or tickers.")
        base_dt = pd.Timestamp(hist.index[0]).tz_localize(None)
        def pick_field(field):
            if isinstance(hist.columns, pd.MultiIndex):
                ser = hist[field].iloc[0]
                ser.index = ser.index.astype(str)
                return pd.Series(ser, dtype="float64").reindex(tickers)
            else:
                return pd.Series({tickers[0]: hist[field].iloc[0]})
    else:
        hist = _yf_download(tickers, period="7d", interval="1d")
        base_dt = pd.Timestamp(hist.index[-1]).tz_localize(None)
        def pick_field(field):
            if isinstance(hist.columns, pd.MultiIndex):
                ser = hist[field].iloc[-1]
                ser.index = ser.index.astype(str)
                return pd.Series(ser, dtype="float64").reindex(tickers)
            else:
                return pd.Series({tickers[0]: hist[field].iloc[-1]})

    base_close = pick_field("Close").fillna(method="ffill")
    base_adj   = pick_field("Adj Close").fillna(method="ffill")

    baseline = {
        "as_of": str(base_dt.date()),
        "capital": NOTIONAL_CAPITAL,
        "base_close": {k: float(base_close.get(k, np.nan)) for k in tickers},
        "base_adj":   {k: float(base_adj.get(k, np.nan)) for k in tickers}
    }
    Path(BASELINE_JSON).write_text(json.dumps(baseline, indent=2))
    return baseline

def compute_weights_from_baseline(port, last_close, base_close):
    """
    port: df with columns ticker,target_w,band_lo,band_hi
    last_close: Series last Close
    base_close: dict ticker->base close
    Returns: alloc df with weight, synthetic $ mkt_value, price column
    """
    tickers = port["ticker"].tolist()
    base_vec = pd.Series({t: base_close.get(t, np.nan) for t in tickers}, dtype="float64")
    last_vec = last_close.reindex(tickers)
    idx = (last_vec / base_vec).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)

    val = NOTIONAL_CAPITAL * port["target_w"].to_numpy(float) * idx.to_numpy(float)
    total_mv = float(np.nansum(val))
    weight = (val / total_mv) if total_mv > 0 else np.zeros_like(val)

    out = port.copy()
    out["price"] = last_vec.values
    out["mkt_value"] = val
    out["weight"] = weight

    def band_status(row):
        w, lo, hi = float(row["weight"]), float(row["band_lo"]), float(row["band_hi"])
        if np.isnan(w) or np.isnan(lo) or np.isnan(hi):
            return "N/A"
        if w < lo - EPS: return "LOW"
        if w > hi + EPS: return "HIGH"
        return "OK"
    out["status"] = out.apply(band_status, axis=1)
    return out, total_mv, idx

def format_email_table(df, title="Current Allocation"):
    lines = [title, "-"*len(title)]
    lines.append(f"{'Ticker':<10}{'Weight':>8}  {'Target':>8}  {'Band':>13}  {'Status':>7}")
    for _, r in df.iterrows():
        band = f"[{r.band_lo*100:.1f}–{r.band_hi*100:.1f}%]"
        lines.append(f"{r.ticker:<10}{r.weight*100:>7.2f}%  {r.target_w*100:>7.1f}%  {band:>13}  {r.status:>7}")
    return "\n".join(lines)

def email_send(subject, body, attachments=None):
    if not all([SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS, MAIL_FROM, MAIL_TO]):
        print("Email not configured; set SMTP_* and MAIL_* secrets.")
        return
    msg = EmailMessage()
    msg["From"] = MAIL_FROM
    msg["To"]   = MAIL_TO
    msg["Subject"] = subject
    msg.set_content(body)
    for att in (attachments or []):
        name, df = att
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        msg.add_attachment(buf.getvalue().encode("utf-8"),
                           maintype="text", subtype="csv", filename=name)
    ctx = ssl.create_default_context()
    with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=30) as s:
        s.starttls(context=ctx)
        s.login(SMTP_USER, SMTP_PASS)
        s.send_message(msg)

def update_history(history_path, dt_label, mv_price, roc_price, mv_tr, roc_tr):
    path = Path(history_path)
    row = f"{dt_label},{mv_price:.2f},{roc_price:.6f},{mv_tr:.2f},{roc_tr:.6f}\n"
    if not path.exists():
        path.write_text("date,mv_price,roc_price,mv_tr,roc_total_return\n" + row)
    else:
        with path.open("a") as f:
            f.write(row)

def load_signals_state(path=SIGNALS_JSON):
    if Path(path).exists():
        try:
            return json.loads(Path(path).read_text())
        except Exception:
            return {}
    return {}

def save_signals_state(state, path=SIGNALS_JSON):
    Path(path).write_text(json.dumps(state, indent=2))

def detect_signals(df_alloc: pd.DataFrame):
    out = []
    for _, r in df_alloc.iterrows():
        if r["status"] in ("LOW","HIGH"):
            out.append({
                "ticker": r["ticker"],
                "status": r["status"],
                "weight": float(r["weight"]),
                "lo": float(r["band_lo"]),
                "hi": float(r["band_hi"]),
                "target": float(r["target_w"])
            })
    return out

def make_rebalance_table(df_alloc, to="target"):
    total_mv = float(df_alloc["mkt_value"].sum())
    if to == "band_mid":
        rebal_target = []
        for _, r in df_alloc.iterrows():
            if not np.isnan(r["band_lo"]) and not np.isnan(r["band_hi"]):
                rebal_target.append(0.5*(r["band_lo"]+r["band_hi"]))
            else:
                rebal_target.append(r["target_w"])
        targ = np.array(rebal_target, float)
    else:
        targ = df_alloc["target_w"].to_numpy(float)
    delta = (targ - df_alloc["weight"].to_numpy(float)) * total_mv
    out = df_alloc[["ticker","price","mkt_value","weight","target_w"]].copy()
    out["rebal_target"] = targ
    out["$delta"] = delta
    return out.sort_values("$delta")

# ============================ Jobs ============================

def daily_job():
    port = load_portfolio()
    tickers = port["ticker"].tolist()

    baseline = load_or_create_baseline(tickers, baseline_date=BASELINE_DATE)
    base_close = baseline["base_close"]
    base_adj   = baseline["base_adj"]
    capital    = float(baseline.get("capital", NOTIONAL_CAPITAL))

    data = _yf_download(tickers, period="7d", interval="1d")
    last_close = _last_series(data, "Close", tickers)
    last_adj   = _last_series(data, "Adj Close", tickers)
    last_dt    = pd.Timestamp(data.index[-1]).tz_localize(None)

    alloc, total_mv_price, idx_price = compute_weights_from_baseline(port, last_close, base_close)

    d2 = _yf_download(tickers, period="3d", interval="1d")
    c_prev, c_now = _two_series(d2, "Close", tickers)
    a_prev, a_now = _two_series(d2, "Adj Close", tickers)

    base_close_ser = pd.Series(base_close, dtype="float64").reindex(tickers)
    base_adj_ser   = pd.Series(base_adj,   dtype="float64").reindex(tickers)
    targ           = port["target_w"].to_numpy(float)

    mv_prev_price = capital * float(np.nansum(targ * (c_prev.to_numpy() / base_close_ser.to_numpy())))
    mv_now_price  = capital * float(np.nansum(targ * (c_now.to_numpy()  / base_close_ser.to_numpy())))
    roc_price     = (mv_now_price / mv_prev_price - 1.0) if mv_prev_price > 0 else 0.0

    mv_prev_tr = capital * float(np.nansum(targ * (a_prev.to_numpy() / base_adj_ser.to_numpy())))
    mv_now_tr  = capital * float(np.nansum(targ * (a_now.to_numpy()  / base_adj_ser.to_numpy())))
    roc_tr     = (mv_now_tr / mv_prev_tr - 1.0) if mv_prev_tr > 0 else 0.0

    signals = detect_signals(alloc)
    prev_state = load_signals_state()
    today_state = { s["ticker"]: s["status"] for s in signals }
    changed = (today_state != prev_state)
    save_signals_state(today_state)

    update_history(HISTORY_CSV, last_dt.date().isoformat(), mv_now_price, roc_price, mv_now_tr, roc_tr)
    snapshot = {
        "as_of": str(last_dt.date()),
        "baseline_date": baseline.get("as_of"),
        "capital": capital,
        "total_mv_price": mv_now_price,
        "roc_price": roc_price,
        "roc_total_return": roc_tr,
        "allocation": alloc[["ticker","price","mkt_value","weight","target_w","band_lo","band_hi","status"]].to_dict(orient="records")
    }
    Path(STATUS_JSON).write_text(json.dumps(snapshot, indent=2))

    if signals and (ALWAYS_ALERT_ON_BREACH or changed):
        header = f"As of {last_dt.date().isoformat()} (EOD)"
        body  = header + "\n\n"
        body += format_email_table(alloc, "Current Allocation") + "\n\n"
        body += "⚠️ Signals (band breaches):\n"
        for s in signals:
            body += f"- {s['ticker']}: {s['status']} (w={s['weight']*100:.2f}% vs band {s['lo']*100:.1f}–{s['hi']*100:.1f}%, target {s['target']*100:.1f}%)\n"
        reb = make_rebalance_table(alloc, to="target")
        email_send(
            subject="Portfolio — SIGNAL alert (bands breached)",
            body=body,
            attachments=[
                ("allocation.csv", alloc[["ticker","price","mkt_value","weight","target_w","band_lo","band_hi","status"]]),
                ("rebalance_to_target.csv", reb),
            ]
        )
    else:
        print("No signal email sent (no breaches or unchanged).")

def weekly_job():
    port = load_portfolio()
    tickers = port["ticker"].tolist()
    baseline = load_or_create_baseline(tickers, baseline_date=BASELINE_DATE)
    base_close = baseline["base_close"]
    base_adj   = baseline["base_adj"]
    capital    = float(baseline.get("capital", NOTIONAL_CAPITAL))

    data = _yf_download(tickers, period="7d", interval="1d")
    last_close = _last_series(data, "Close", tickers)
    last_dt    = pd.Timestamp(data.index[-1]).tz_localize(None)

    alloc, total_mv_price, idx_price = compute_weights_from_baseline(port, last_close, base_close)

    hist = None
    if Path(HISTORY_CSV).exists():
        hist = pd.read_csv(HISTORY_CSV, parse_dates=["date"]).sort_values("date")

    def span_return(colname, days):
        if hist is None or len(hist) < 2:
            return np.nan
        end = hist[colname].iloc[-1]
        start_idx = max(0, len(hist)-1-days)
        start = hist[colname].iloc[start_idx]
        return (end / start - 1.0) if start else np.nan

    ret_1w  = span_return("mv_tr", 5)
    ret_mtd = np.nan
    if hist is not None and len(hist):
        this_month = hist[hist["date"].dt.to_period("M") == hist["date"].iloc[-1].to_period("M")]
        if len(this_month) >= 2:
            ret_mtd = this_month["mv_tr"].iloc[-1] / this_month["mv_tr"].iloc[0] - 1.0
    ret_ytd = np.nan
    if hist is not None and len(hist):
        this_year = hist[hist["date"].dt.year == hist["date"].iloc[-1].year]
        if len(this_year) >= 2:
            ret_ytd = this_year["mv_tr"].iloc[-1] / this_year["mv_tr"].iloc[0] - 1.0

    header = f"Weekly recap — as of {last_dt.date().isoformat()} (baseline {baseline.get('as_of')})"
    body  = header + "\n\n"
    body += format_email_table(alloc, "Current Allocation") + "\n\n"
    body += "Performance (Adj Close proxy for TR):\n"
    body += (f"- 1w:  {ret_1w*100:6.2f}%\n" if np.isfinite(ret_1w) else "- 1w:   n/a\n")
    body += (f"- MTD: {ret_mtd*100:6.2f}%\n" if np.isfinite(ret_mtd) else "- MTD:  n/a\n")
    body += (f"- YTD: {ret_ytd*100:6.2f}%\n" if np.isfinite(ret_ytd) else "- YTD:  n/a\n")

    reb_target = make_rebalance_table(alloc, to="target")
    reb_mid    = make_rebalance_table(alloc, to="band_mid")

    email_send(
        subject="Portfolio — Weekly recap",
        body=body,
        attachments=[
            ("allocation.csv", alloc[["ticker","price","mkt_value","weight","target_w","band_lo","band_hi","status"]]),
            ("rebalance_to_target.csv", reb_target),
            ("rebalance_to_band_mid.csv", reb_mid),
            ("history.csv", pd.read_csv(HISTORY_CSV) if Path(HISTORY_CSV).exists() else pd.DataFrame()),
        ]
    )

# ============================ CLI ============================

if __name__ == "__main__":
    mode = (sys.argv[1] if len(sys.argv) > 1 else "daily").lower()
    if mode == "daily":
        daily_job()
    elif mode == "weekly":
        weekly_job()
    else:
        print("Usage: python tracker.py [daily|weekly]")
        sys.exit(2)
