#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Portfolio tracker:
- Daily EOD: compute weights vs target, detect band-breach "signals", email IF any signals.
- Weekly: full recap (weights, breaches, simple performance).
Outputs:
- history.csv (daily MV + price/TR returns)
- status.json (latest snapshot)
- signals_state.json (to detect new/cleared breaches)
"""

import os, io, json, smtplib, ssl, sys
from email.message import EmailMessage
from pathlib import Path
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd
import numpy as np
import yfinance as yf

# ---------- Config ----------
PORTFOLIO_CSV = "portfolio.csv"
HISTORY_CSV   = "history.csv"
STATUS_JSON   = "status.json"
SIGNALS_JSON  = "signals_state.json"

TZ_NY   = ZoneInfo("America/New_York")   # market close reference
TZ_USER = ZoneInfo("America/Denver")     # your local
EPS     = 1e-6                           # band epsilon

SMTP_HOST = os.environ.get("SMTP_HOST")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
SMTP_USER = os.environ.get("SMTP_USER")
SMTP_PASS = os.environ.get("SMTP_PASS")
MAIL_FROM = os.environ.get("MAIL_FROM")
MAIL_TO   = os.environ.get("MAIL_TO")

# ---------- Helpers ----------
def now_local():
    return datetime.now(TZ_USER)

def today_trading_date_close():
    """Return the latest available daily bar date we should use.
       We’ll use the last row returned by yfinance daily."""
    return None  # We infer from downloaded data

def load_portfolio(path=PORTFOLIO_CSV) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = {"ticker","shares","target_w","band_lo","band_hi"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"portfolio.csv missing columns: {sorted(missing)}")
    df["ticker"]   = df["ticker"].astype(str)
    df["shares"]   = pd.to_numeric(df["shares"], errors="coerce").fillna(0.0)
    for c in ["target_w","band_lo","band_hi"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # normalize targets if they don't sum to 1
    tw = df["target_w"].to_numpy(float)
    if not np.isclose(tw.sum(), 1.0):
        tw = tw / np.nansum(tw)
        df["target_w"] = tw
    return df

def fetch_prices(tickers):
    """
    Returns:
      close: pd.Series last available Close
      adj:   pd.Series last available Adj Close (dividends reinvested proxy)
      last_dt: timestamp of that last bar (NY time naive date)
    """
    # Pull a week to be safe; sometimes day-of bar appears with slight delay
    data = yf.download(
        tickers=tickers, period="7d", interval="1d",
        auto_adjust=False, progress=False, group_by='column'
    )

    # yfinance returns different shapes for 1 vs many tickers; unify
    def last_col(frame, field):
        if isinstance(frame.columns, pd.MultiIndex):
            # wide: top level fields: ('Adj Close', 'AAPL') etc
            ser = frame[field].iloc[-1]
            ser.index = ser.index.astype(str)
            return pd.Series(ser, dtype="float64")
        else:
            # single ticker: frame[field] is a Series
            return pd.Series({tickers[0]: frame[field].iloc[-1]})

    last_close = last_col(data, "Close")
    last_adj   = last_col(data, "Adj Close")
    # Extract the last date (index is DatetimeIndex in UTC or local?)
    if isinstance(data.index, pd.DatetimeIndex) and len(data.index):
        last_dt = data.index[-1].tz_localize(None)
    else:
        last_dt = pd.Timestamp.utcnow().tz_localize(None)

    return last_close, last_adj, last_dt

def compute_weights(df: pd.DataFrame, price_ser: pd.Series):
    df2 = df.copy()
    df2 = df2.merge(price_ser.rename("price"), left_on="ticker", right_index=True, how="left")
    df2["mkt_value"] = df2["shares"] * df2["price"]
    total_mv = float(df2["mkt_value"].sum())
    df2["weight"] = df2["mkt_value"] / (total_mv if total_mv > 0 else np.nan)
    # band status
    def band_status(row):
        w = float(row["weight"])
        lo = float(row["band_lo"])
        hi = float(row["band_hi"])
        if np.isnan(w) or np.isnan(lo) or np.isnan(hi):
            return "N/A"
        if w < lo - EPS: return "LOW"
        if w > hi + EPS: return "HIGH"
        return "OK"
    df2["status"] = df2.apply(band_status, axis=1)
    return df2, total_mv

def format_email_table(df, title="Current Allocation"):
    # make a tidy monospace text table (simple + robust)
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
    # optional attachments
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

def update_history(history_path, dt_label, mv_price, roc_price, tr_price, roc_tr):
    path = Path(history_path)
    row = f"{dt_label},{mv_price:.2f},{roc_price:.6f},{tr_price:.2f},{roc_tr:.6f}\n"
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
    """Return list of dicts for assets currently out of band."""
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
    """Suggest $ to move to reach targets ('target') or band midpoints ('band_mid')."""
    total_mv = float(df_alloc["mkt_value"].sum())
    if to == "band_mid":
        targ = []
        for _, r in df_alloc.iterrows():
            if not np.isnan(r["band_lo"]) and not np.isnan(r["band_hi"]):
                targ.append((r["band_lo"] + r["band_hi"]) / 2.0)
            else:
                targ.append(r["target_w"])
        df_alloc = df_alloc.assign(rebal_target=targ)
    else:
        df_alloc = df_alloc.assign(rebal_target=df_alloc["target_w"])
    df_alloc["$delta"] = (df_alloc["rebal_target"] - df_alloc["weight"]) * total_mv
    out = df_alloc[["ticker","weight","target_w","rebal_target","$delta","status","price"]].copy()
    out = out.sort_values("$delta")
    return out

def daily_job():
    # 1) Load portfolio
    port = load_portfolio()

    # 2) Fetch last Close & Adj Close
    close, adj, last_dt = fetch_prices(port["ticker"].tolist())

    # 3) Compute allocation & band status
    alloc, total_mv = compute_weights(port, close)

    # 4) Compute daily RoC (price & TR) relative to the previous bar
    #    (We fetch 7d, but only the *last two* matter for ROC. Easiest: refetch two rows quickly.)
    data = yf.download(
        tickers=port["ticker"].tolist(), period="3d", interval="1d",
        auto_adjust=False, progress=False, group_by='column'
    )
    # unify again
    def pick_two(field):
        if isinstance(data.columns, pd.MultiIndex):
            ser_prev = data[field].iloc[-2]
            ser_now  = data[field].iloc[-1]
            return ser_prev, ser_now
        else:
            # single symbol
            return pd.Series({port["ticker"].iloc[0]: data[field].iloc[-2]}), pd.Series({port["ticker"].iloc[0]: data[field].iloc[-1]})
    c_prev, c_now = pick_two("Close")
    a_prev, a_now = pick_two("Adj Close")

    mv_prev = float((port.set_index("ticker")["shares"] * c_prev).sum())
    mv_now  = float((port.set_index("ticker")["shares"] * c_now).sum())
    roc_price = (mv_now / mv_prev - 1.0) if mv_prev else 0.0

    tr_prev = float((port.set_index("ticker")["shares"] * a_prev).sum())
    tr_now  = float((port.set_index("ticker")["shares"] * a_now).sum())
    roc_tr = (tr_now / tr_prev - 1.0) if tr_prev else 0.0

    # 5) Signals
    signals = detect_signals(alloc)
    prev_state = load_signals_state()
    # Build today's simple state
    today_state = { s["ticker"]: s["status"] for s in signals }
    # Determine if new signals appeared or cleared
    changed = (today_state != prev_state)

    # 6) Save artifacts
    #    a) history.csv (append)
    label = (last_dt.date().isoformat() if isinstance(last_dt, pd.Timestamp) else datetime.now().date().isoformat())
    update_history(HISTORY_CSV, label, mv_now, roc_price, tr_now, roc_tr)
    #    b) status.json
    snapshot = {
        "as_of": label,
        "total_mv": total_mv,
        "roc_price": roc_price,
        "roc_total_return": roc_tr,
        "allocation": alloc[["ticker","shares","price","mkt_value","weight","target_w","band_lo","band_hi","status"]].to_dict(orient="records")
    }
    Path(STATUS_JSON).write_text(json.dumps(snapshot, indent=2))
    #    c) signals_state.json
    save_signals_state(today_state)

    # 7) Email if there are ANY current-day signals (breaches) — default is “on change” to reduce noise
    ALWAYS_ALERT_ON_BREACH = False  # flip to True if you want a daily alert any time breaches exist
    if (signals and (ALWAYS_ALERT_ON_BREACH or changed)):
        header = f"As of {label} (NY close)"
        body  = header + "\n\n"
        body += format_email_table(alloc, "Current Allocation") + "\n\n"
        body += "⚠️ Signals (band breaches):\n"
        for s in signals:
            body += f"- {s['ticker']}: {s['status']} (w={s['weight']*100:.2f}% vs band {s['lo']*100:.1f}–{s['hi']*100:.1f}%, target {s['target']*100:.1f}%)\n"
        # Optional suggestions
        reb = make_rebalance_table(alloc, to="target")
        email_send(
            subject="Portfolio — SIGNAL alert (bands breached)",
            body=body,
            attachments=[("allocation.csv", alloc[["ticker","shares","price","mkt_value","weight","target_w","band_lo","band_hi","status"]]),
                         ("rebalance_to_target.csv", reb)]
        )
    else:
        print("No signal email sent (no breaches or unchanged).")

def weekly_job():
    port = load_portfolio()
    close, adj, last_dt = fetch_prices(port["ticker"].tolist())
    alloc, total_mv = compute_weights(port, close)

    # Read history to compute 1w/MTD/YTD quickly
    hist = None
    if Path(HISTORY_CSV).exists():
        hist = pd.read_csv(HISTORY_CSV, parse_dates=["date"])
        hist = hist.sort_values("date")
    # Simple returns from history
    def span_return(colname, days):
        if hist is None or len(hist) < 2:
            return np.nan
        end = hist[colname].iloc[-1]
        start_idx = max(0, len(hist)-1-days)
        start = hist[colname].iloc[start_idx]
        return (end / start - 1.0) if start else np.nan
    # 5 trading days ~ 1w (approx; we log once/day)
    ret_1w  = span_return("mv_tr", 5)
    # MTD
    ret_mtd = np.nan
    if hist is not None:
        this_month = hist[hist["date"].dt.to_period("M") == hist["date"].iloc[-1].to_period("M")]
        if len(this_month) >= 2:
            ret_mtd = this_month["mv_tr"].iloc[-1] / this_month["mv_tr"].iloc[0] - 1.0
    # YTD
    ret_ytd = np.nan
    if hist is not None:
        this_year = hist[hist["date"].dt.year == hist["date"].iloc[-1].year]
        if len(this_year) >= 2:
            ret_ytd = this_year["mv_tr"].iloc[-1] / this_year["mv_tr"].iloc[0] - 1.0

    header = f"Weekly recap — as of {last_dt.date().isoformat()} (NY close)"
    body  = header + "\n\n"
    body += format_email_table(alloc, "Current Allocation") + "\n\n"
    body += "Performance (Adj Close proxy for TR):\n"
    body += f"- 1w:  {ret_1w*100:6.2f}%\n" if np.isfinite(ret_1w) else "- 1w:   n/a\n"
    body += f"- MTD: {ret_mtd*100:6.2f}%\n" if np.isfinite(ret_mtd) else "- MTD:  n/a\n"
    body += f"- YTD: {ret_ytd*100:6.2f}%\n" if np.isfinite(ret_ytd) else "- YTD:  n/a\n"

    reb_target = make_rebalance_table(alloc, to="target")
    reb_mid    = make_rebalance_table(alloc, to="band_mid")

    email_send(
        subject="Portfolio — Weekly recap",
        body=body,
        attachments=[
            ("allocation.csv", alloc[["ticker","shares","price","mkt_value","weight","target_w","band_lo","band_hi","status"]]),
            ("rebalance_to_target.csv", reb_target),
            ("rebalance_to_band_mid.csv", reb_mid),
            ("history.csv", pd.read_csv(HISTORY_CSV) if Path(HISTORY_CSV).exists() else pd.DataFrame())
        ]
    )

# ---------- CLI ----------
if __name__ == "__main__":
    mode = (sys.argv[1] if len(sys.argv) > 1 else "daily").lower()
    if mode == "daily":
        daily_job()
    elif mode == "weekly":
        weekly_job()
    else:
        print("Usage: python tracker.py [daily|weekly]")
        sys.exit(2)
