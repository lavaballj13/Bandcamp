#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Weights-only portfolio tracker (hardened)
- No real shares required. We track drift vs targets using baseline prices.
- First run creates baseline.json with baseline Close & Adj Close per ticker (with fallbacks).
- Daily EOD: compute current weights, detect band breaches ("signals"), email if signals appear/clear.
- Weekly: recap email with performance and suggested $ deltas to reach targets.

Artifacts:
- baseline.json        (created on first run if missing)
- history.csv          (daily MV logs; price & TR proxy)
- status.json          (latest snapshot)
- signals_state.json   (to detect "state changed" signals)

Environment toggles (optional):
- BASELINE_DATE        : override in code or via env; 'YYYY-MM-DD' or '' to use last bar on first run
- EMAIL_DISABLE=1      : skip sending email (useful during debugging)
- TRACKER_DEBUG=1      : verbose logging of inputs and computed values
- ALWAYS_ALERT_ON_BREACH=1 : force daily emails any day breaches exist (not only when state changes)
"""

import os, io, json, smtplib, ssl, sys, math, time
from email.message import EmailMessage
from pathlib import Path
import pandas as pd
import numpy as np
import yfinance as yf

# ---------- User-configurable knobs ----------
NOTIONAL_CAPITAL = float(os.getenv("NOTIONAL_CAPITAL", "1000.0"))  # $ for synthetic MV
BASELINE_DATE    = os.getenv("BASELINE_DATE", "2025-07-17")        # "" => set on first run (latest bar)
ALWAYS_ALERT_ON_BREACH = (os.getenv("ALWAYS_ALERT_ON_BREACH", "0") == "1")

# ---------- Files ----------
PORTFOLIO_CSV = "portfolio.csv"
BASELINE_JSON = "baseline.json"
HISTORY_CSV   = "history.csv"
STATUS_JSON   = "status.json"
SIGNALS_JSON  = "signals_state.json"

# ---------- Email (from GitHub Secrets) ----------
SMTP_HOST = os.getenv("SMTP_HOST")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASS = os.getenv("SMTP_PASS")
MAIL_FROM = os.getenv("MAIL_FROM")
MAIL_TO   = os.getenv("MAIL_TO")
EMAIL_DISABLE = (os.getenv("EMAIL_DISABLE", "0") == "1")

# ---------- Misc ----------
EPS = 1e-9
DBG = (os.getenv("TRACKER_DEBUG", "0") == "1")

# ============================ Utilities ============================

def _dbg(*a):
    if DBG:
        print("[dbg]", *a, flush=True)

def _warn(*a):
    print("[warn]", *a, flush=True)

def _err(*a):
    print("[error]", *a, flush=True)

def _collapse_history_df(df: pd.DataFrame) -> pd.DataFrame:
    """Sort by date, then keep the LAST row per calendar date (idempotent)."""
    if df is None or len(df) == 0:
        return df
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")
    # group by calendar day, keep last occurrence
    g = df.groupby(df["date"].dt.date, as_index=False).last()
    g["date"] = pd.to_datetime(g["date"])
    return g

def _load_history_collapsed(path=HISTORY_CSV) -> pd.DataFrame | None:
    if not Path(path).exists():
        return None
    df = pd.read_csv(path, parse_dates=["date"])
    return _collapse_history_df(df)

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
    if not np.isclose(np.nansum(tw), 1.0) and np.nansum(tw) > 0:
        tw = tw / np.nansum(tw)
        df["target_w"] = tw
    if DBG:
        _dbg("portfolio.csv loaded:\n", df.to_string(index=False))
    return df

# --- Robust yfinance helpers ---

def _yf_download(tickers, period="7d", interval="1d", tries=3):
    """Robust yfinance download with small retries; returns a DataFrame (possibly empty)."""
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
    _warn(f"yfinance download empty after {tries} tries (period={period}, interval={interval}) err={last_err}")
    return pd.DataFrame()

def _get_field_safe(frame, field: str):
    """Return a DataFrame for the requested field; if missing, try sensible fallbacks."""
    if not isinstance(frame, pd.DataFrame) or len(frame) == 0:
        raise KeyError("empty frame")
    if isinstance(frame.columns, pd.MultiIndex):
        # Try exact field
        try:
            return frame[field]
        except Exception:
            pass
        # Fallbacks
        if field == "Adj Close":
            try:
                _warn("Adj Close missing; falling back to Close for TR proxy")
                return frame["Close"]
            except Exception:
                pass
        raise KeyError(f"Field {field!r} not found in MultiIndex columns: {list(set([lvl[0] for lvl in frame.columns]))[:5]}...")
    else:
        # Single-ticker frames expose columns like ['Open','High','Low','Close','Adj Close',...]
        if field in frame.columns:
            return frame[[field]]
        if field == "Adj Close" and "Close" in frame.columns:
            _warn("Adj Close missing (single ticker); falling back to Close")
            return frame[["Close"]]
        raise KeyError(f"Field {field!r} not found in columns: {frame.columns.tolist()}")

def _last_series(frame, field, tickers):
    """Get last-row Series for field; if empty, return NaNs indexed by tickers (with fallback if needed)."""
    try:
        sub = _get_field_safe(frame, field)
    except Exception:
        # return NaNs for safety
        return pd.Series({t: np.nan for t in tickers}, dtype="float64")

    if isinstance(frame.columns, pd.MultiIndex):
        ser = sub.iloc[-1]
        ser.index = ser.index.astype(str)
        return pd.Series(ser, dtype="float64").reindex(tickers)
    else:
        # single ticker case => 'sub' still a DataFrame with one column
        val = float(sub.iloc[-1, 0])
        return pd.Series({tickers[0]: val}, dtype="float64")

def _two_series(frame, field, tickers):
    """Return (prev, now) Series for field; if <2 rows, duplicate the last; if empty, return NaNs."""
    if not isinstance(frame, pd.DataFrame) or len(frame) == 0:
        s = pd.Series({t: np.nan for t in tickers}, dtype="float64")
        return s, s
    if len(frame) == 1:
        s_last = _last_series(frame, field, tickers)
        return s_last, s_last
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
        return (pd.Series({tickers[0]: float(sub.iloc[-2,0])}, dtype="float64"),
                pd.Series({tickers[0]: float(sub.iloc[-1,0])}, dtype="float64"))

# --- Baseline handling ---

def _pick_baseline_row(hist: pd.DataFrame, tickers) -> tuple[pd.Timestamp, pd.Series, pd.Series]:
    """Pick the baseline timestamp and base Close/Adj values from a yfinance frame."""
    if not isinstance(hist, pd.DataFrame) or len(hist) == 0:
        raise RuntimeError("Empty history when creating baseline")
    # Choose first available row in the window
    base_dt = pd.Timestamp(hist.index[0]).tz_localize(None)

    # Extract first row Close / Adj Close (fallback to Close if Adj Close absent)
    close_df = _get_field_safe(hist, "Close")
    try:
        adj_df = _get_field_safe(hist, "Adj Close")
    except Exception:
        adj_df = close_df  # fallback already warned in _get_field_safe

    if isinstance(hist.columns, pd.MultiIndex):
        base_close = close_df.iloc[0]; base_close.index = base_close.index.astype(str)
        base_adj   = adj_df.iloc[0];   base_adj.index   = base_adj.index.astype(str)
        base_close = pd.Series(base_close, dtype="float64").reindex(tickers)
        base_adj   = pd.Series(base_adj,   dtype="float64").reindex(tickers)
    else:
        base_close = pd.Series({tickers[0]: float(close_df.iloc[0,0])}, dtype="float64")
        base_adj   = pd.Series({tickers[0]: float(adj_df.iloc[0,0])}, dtype="float64")

    return base_dt, base_close, base_adj

def load_or_create_baseline(tickers, baseline_date=BASELINE_DATE):
    """Load baseline if exists; else create using baseline_date (or last bar if blank)."""
    if Path(BASELINE_JSON).exists():
        try:
            js = json.loads(Path(BASELINE_JSON).read_text())
            if DBG: _dbg("baseline.json loaded:", js)
            return js
        except Exception as e:
            _warn(f"baseline.json unreadable, recreating: {e}")

    if baseline_date:
        # Try to start at that date; if nothing, widen a bit
        windows = [
            dict(start=baseline_date, end=None),        # as requested
            dict(start=baseline_date, end=None),        # retry once
        ]
        hist = None
        for w in windows:
            hist = yf.download(tickers=tickers, start=w["start"], end=w["end"],
                               interval="1d", auto_adjust=False, progress=False, group_by="column")
            if isinstance(hist, pd.DataFrame) and len(hist) >= 1:
                break
            time.sleep(0.5)
        if not isinstance(hist, pd.DataFrame) or len(hist) == 0:
            # Final fallback: take recent 60d and pick first row >= baseline_date if present
            hist = yf.download(tickers=tickers, period="60d", interval="1d",
                               auto_adjust=False, progress=False, group_by="column")
            if not isinstance(hist.index, pd.DatetimeIndex) or len(hist) == 0:
                raise RuntimeError("Could not fetch data to create baseline; check BASELINE_DATE or tickers.")
            # If we have dates, try to choose >= baseline_date; else first row
            try:
                ix = hist.index.get_indexer([pd.to_datetime(baseline_date)], method="backfill")
                row0 = int(ix[0]) if ix.size else 0
                hist = hist.iloc[row0:]
            except Exception:
                pass
        base_dt, base_close, base_adj = _pick_baseline_row(hist, tickers)
    else:
        hist = _yf_download(tickers, period="7d", interval="1d", tries=3)
        if not isinstance(hist, pd.DataFrame) or len(hist) == 0:
            raise RuntimeError("Could not fetch recent data to create baseline")
        # Use the last bar as baseline
        if isinstance(hist.index, pd.DatetimeIndex):
            hist = hist.iloc[[-1]]
        base_dt, base_close, base_adj = _pick_baseline_row(hist, tickers)

    baseline = {
        "as_of": str(base_dt.date()),
        "capital": NOTIONAL_CAPITAL,
        "base_close": {k: float(base_close.get(k, np.nan)) for k in tickers},
        "base_adj":   {k: float(base_adj.get(k, np.nan)) for k in tickers}
    }
    Path(BASELINE_JSON).write_text(json.dumps(baseline, indent=2))
    if DBG: _dbg("baseline.json created:", baseline)
    return baseline

# --- Core math ---

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
    if DBG: _dbg("allocation table:\n", out.to_string(index=False))
    return out, total_mv, idx

def format_email_table(df, title="Current Allocation"):
    lines = [title, "-"*len(title)]
    lines.append(f"{'Ticker':<10}{'Weight':>8}  {'Target':>8}  {'Band':>13}  {'Status':>7}")
    for _, r in df.iterrows():
        band = f"[{r.band_lo*100:.1f}–{r.band_hi*100:.1f}%]"
        lines.append(f"{r.ticker:<10}{r.weight*100:>7.2f}%  {r.target_w*100:>7.1f}%  {band:>13}  {r.status:>7}")
    return "\n".join(lines)

def email_send(subject, body, attachments=None):
    if EMAIL_DISABLE:
        _warn("EMAIL_DISABLE=1 → skipping email send")
        return
    if not all([SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS, MAIL_FROM, MAIL_TO]):
        _warn("Email not configured; set SMTP_* and MAIL_* secrets.")
        return
    try:
        msg = EmailMessage()
        msg["From"] = MAIL_FROM
        msg["To"]   = MAIL_TO
        msg["Subject"] = subject
        msg.set_content(body)
        for att in (attachments or []):
            name, df = att
            buf = io.StringIO(); df.to_csv(buf, index=False)
            msg.add_attachment(buf.getvalue().encode("utf-8"),
                               maintype="text", subtype="csv", filename=name)
        ctx = ssl.create_default_context()
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=30) as s:
            s.starttls(context=ctx)
            s.login(SMTP_USER, SMTP_PASS)
            s.send_message(msg)
        print("[info] email sent:", subject, flush=True)
    except Exception as e:
        _warn(f"email_send failed: {repr(e)}")

def update_history(history_path, dt_label, mv_price, roc_price, mv_tr, roc_tr):
    """
    Upsert: ensure only one row per calendar date (keep the latest).
    """
    path = Path(history_path)
    cols = ["date", "mv_price", "roc_price", "mv_tr", "roc_total_return"]

    # build the new row (date only; no time) for idempotency
    day = pd.to_datetime(str(dt_label)).date().isoformat()
    new_row = pd.DataFrame([{
        "date": day,
        "mv_price": float(mv_price),
        "roc_price": float(roc_price),
        "mv_tr": float(mv_tr),
        "roc_total_return": float(roc_tr),
    }], columns=cols)

    if path.exists():
        try:
            df = pd.read_csv(path, parse_dates=["date"])
        except Exception:
            df = pd.DataFrame(columns=cols)
        # collapse any existing dup dates, then drop today's if present
        df = _collapse_history_df(df)
        if len(df):
            df = df[df["date"].dt.date.astype(str) != day]
        df = pd.concat([df, new_row], ignore_index=True)
    else:
        df = new_row

    # final ordering / formatting
    df = df.sort_values("date")
    df_out = df.copy()
    df_out["date"] = df_out["date"].dt.strftime("%Y-%m-%d")
    df_out.to_csv(path, index=False)

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
    try:
        port = load_portfolio()
        tickers = port["ticker"].tolist()

        baseline = load_or_create_baseline(tickers, baseline_date=BASELINE_DATE)
        base_close = baseline["base_close"]
        base_adj   = baseline["base_adj"]
        capital    = float(baseline.get("capital", NOTIONAL_CAPITAL))

        data = _yf_download(tickers, period="7d", interval="1d", tries=3)
        last_close = _last_series(data, "Close", tickers)
        last_adj   = _last_series(data, "Adj Close", tickers)
        if isinstance(data.index, pd.DatetimeIndex) and len(data.index) >= 1:
            last_dt = pd.Timestamp(data.index[-1]).tz_localize(None)
        else:
            _warn("empty/invalid data index; using UTC now")
            last_dt = pd.Timestamp.utcnow().tz_localize(None)

        alloc, total_mv_price, idx_price = compute_weights_from_baseline(port, last_close, base_close)

        d2 = _yf_download(tickers, period="3d", interval="1d", tries=3)
        c_prev, c_now = _two_series(d2, "Close", tickers)
        a_prev, a_now = _two_series(d2, "Adj Close", tickers)

        base_close_ser = pd.Series(base_close, dtype="float64").reindex(tickers)
        base_adj_ser   = pd.Series(base_adj,   dtype="float64").reindex(tickers)
        targ           = port["target_w"].to_numpy(float)

        # Synthetic MV using baseline indexes
        mv_prev_price = float(NOTIONAL_CAPITAL * np.nansum(targ * (c_prev.to_numpy() / base_close_ser.to_numpy())))
        mv_now_price  = float(NOTIONAL_CAPITAL * np.nansum(targ * (c_now.to_numpy()  / base_close_ser.to_numpy())))
        roc_price     = (mv_now_price / mv_prev_price - 1.0) if mv_prev_price > 0 else 0.0

        mv_prev_tr = float(NOTIONAL_CAPITAL * np.nansum(targ * (a_prev.to_numpy() / base_adj_ser.to_numpy())))
        mv_now_tr  = float(NOTIONAL_CAPITAL * np.nansum(targ * (a_now.to_numpy()  / base_adj_ser.to_numpy())))
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
            print("No signal email sent (no breaches or unchanged).", flush=True)

    except Exception as e:
        _err("daily_job crashed:", repr(e))

def weekly_job():
    try:
        port = load_portfolio()
        tickers = port["ticker"].tolist()
        baseline = load_or_create_baseline(tickers, baseline_date=BASELINE_DATE)
        base_close = baseline["base_close"]
        base_adj   = baseline["base_adj"]
        capital    = float(baseline.get("capital", NOTIONAL_CAPITAL))

        data = _yf_download(tickers, period="7d", interval="1d", tries=3)
        last_close = _last_series(data, "Close", tickers)
        if isinstance(data.index, pd.DatetimeIndex) and len(data.index) >= 1:
            last_dt = pd.Timestamp(data.index[-1]).tz_localize(None)
        else:
            _warn("empty/invalid data index; using UTC now")
            last_dt = pd.Timestamp.utcnow().tz_localize(None)

        alloc, total_mv_price, idx_price = compute_weights_from_baseline(port, last_close, base_close)

        # Read history for basic spans
        # Load history collapsed to one row per calendar date (keep latest)
        hist = None
        try:
            hist = _load_history_collapsed(HISTORY_CSV)
        except Exception as e:
            _warn(f"failed to read/collapse history.csv: {e}")


        def span_return(colname, days):
            if hist is None or len(hist) < 2:
                return np.nan
            end = hist[colname].iloc[-1]
            start_idx = max(0, len(hist)-1-days)
            start = hist[colname].iloc[start_idx]
            return (end / start - 1.0) if start else np.nan

        ret_1w  = span_return("mv_tr", 5)
        # MTD
        ret_mtd = np.nan
        if hist is not None and len(hist):
            this_month = hist[hist["date"].dt.to_period("M") == hist["date"].iloc[-1].to_period("M")]
            if len(this_month) >= 2:
                ret_mtd = this_month["mv_tr"].iloc[-1] / this_month["mv_tr"].iloc[0] - 1.0
        # YTD
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

    except Exception as e:
        _err("weekly_job crashed:", repr(e))

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
