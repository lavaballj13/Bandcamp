        base_dt = pd.Timestamp(hist.index[-1]).tz_localize(None)
        hist = hist.iloc[[-1]]

    close_df = _get_field_safe(hist, "Close")
    try:
        adj_df = _get_field_safe(hist, "Adj Close")
    except Exception:
        adj_df = close_df

    if isinstance(hist.columns, pd.MultiIndex):
        base_close = close_df.iloc[0]; base_close.index = base_close.index.astype(str)
        base_adj   = adj_df.iloc[0];   base_adj.index   = base_adj.index.astype(str)
        base_close = pd.Series(base_close, dtype="float64").reindex(tickers)
        base_adj   = pd.Series(base_adj,   dtype="float64").reindex(tickers)
    else:
        base_close = pd.Series({tickers[0]: float(close_df.iloc[0,0])}, dtype="float64")
        base_adj   = pd.Series({tickers[0]: float(adj_df.iloc[0,0])}, dtype="float64")

    baseline = {
        "as_of": str(base_dt.date()),
        "capital": NOTIONAL_CAPITAL,
        "base_close": {k: float(base_close.get(k, np.nan)) for k in tickers},
        "base_adj":   {k: float(base_adj.get(k, np.nan)) for k in tickers},
    }
    Path(BASELINE_JSON).write_text(json.dumps(baseline, indent=2))
    return baseline

def load_or_init_holdings(port: pd.DataFrame, baseline: dict) -> dict:
    """
    holdings_state.json = {"as_of": "...", "cash": 0.0, "shares": {"TICK": float, ...}}
    If missing, initialize shares using baseline Close and target weights.
    """
    if Path(HOLDINGS_JSON).exists():
        try:
            h = json.loads(Path(HOLDINGS_JSON).read_text())
            if "shares" in h and isinstance(h["shares"], dict):
                return h
        except Exception as e:
            _warn(f"holdings_state.json unreadable, re-init: {e}")

    tickers = port["ticker"].tolist()
    base_close = pd.Series(baseline["base_close"], dtype="float64").reindex(tickers)
    capital = float(baseline.get("capital", NOTIONAL_CAPITAL))
    targ = port["target_w"].to_numpy(float)

    # shares = (capital * target_w) / baseline_price
    shares = {}
    for i, t in enumerate(tickers):
        px = float(base_close.get(t, np.nan))
        if not np.isfinite(px) or px <= 0:
            shares[t] = 0.0
        else:
            shares[t] = float((capital * targ[i]) / px)

    h = {"as_of": baseline.get("as_of"), "cash": 0.0, "shares": shares}
    Path(HOLDINGS_JSON).write_text(json.dumps(h, indent=2))
    return h

def save_holdings(h: dict):
    Path(HOLDINGS_JSON).write_text(json.dumps(h, indent=2))

def load_rebalance_state() -> dict:
    """
    rebalance_state.json =
      {
        "last_rebalance_date": "YYYY-MM-DD" or null,
        "locked_until": "YYYY-MM-DD" or null,
        "lockup_days": 365,
        "suppressed_signals": 0,
        "rebalances": 0
      }
    """
    if Path(REBAL_STATE_JSON).exists():
        try:
            s = json.loads(Path(REBAL_STATE_JSON).read_text())
            if isinstance(s, dict):
                s.setdefault("lockup_days", LOCKUP_DAYS)
                s.setdefault("suppressed_signals", 0)
                s.setdefault("rebalances", 0)
                return s
        except Exception:
            pass
    s = {
        "last_rebalance_date": None,
        "locked_until": None,
        "lockup_days": LOCKUP_DAYS,
        "suppressed_signals": 0,
        "rebalances": 0,
    }
    Path(REBAL_STATE_JSON).write_text(json.dumps(s, indent=2))
    return s

def save_rebalance_state(s: dict):
    Path(REBAL_STATE_JSON).write_text(json.dumps(s, indent=2))

def in_lockup(as_of: date, s: dict) -> tuple[bool, int]:
    locked_until = _parse_date(s.get("locked_until"))
    if locked_until is None:
        return False, 0
    # lockup active if as_of < locked_until
    active = as_of < locked_until
    remaining = (locked_until - as_of).days if active else 0
    return active, max(0, int(remaining))

def set_lockup(as_of: date, s: dict):
    s["last_rebalance_date"] = as_of.isoformat()
    s["locked_until"] = (as_of + timedelta(days=int(s.get("lockup_days", LOCKUP_DAYS)))).isoformat()
    s["rebalances"] = int(s.get("rebalances", 0)) + 1

# ============================ Core math ============================

def compute_allocation_from_holdings(port: pd.DataFrame, last_close: pd.Series, holdings: dict):
    """
    Uses synthetic shares to compute value + weights.
    """
    tickers = port["ticker"].tolist()
    shares = pd.Series(holdings.get("shares", {}), dtype="float64").reindex(tickers).fillna(0.0)
    cash = float(holdings.get("cash", 0.0))

    px = last_close.reindex(tickers).astype(float)
    val = (shares.to_numpy(float) * px.to_numpy(float))
    total = float(np.nansum(val) + cash)
    weight = (val / total) if total > 0 else np.zeros_like(val)

    out = port.copy()
    out["price"] = px.values
    out["shares"] = shares.values
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
    return out, total

def rebalance_holdings_to_target(port: pd.DataFrame, last_close: pd.Series, holdings: dict) -> dict:
    """
    Set shares so weights == targets at current prices (synthetic rebalance).
    """
    tickers = port["ticker"].tolist()
    px = last_close.reindex(tickers).astype(float)
    alloc, total = compute_allocation_from_holdings(port, last_close, holdings)

    targ = port["target_w"].to_numpy(float)
    new_shares = {}
    for i, t in enumerate(tickers):
        p = float(px.iloc[i])
        if not np.isfinite(p) or p <= 0:
            new_shares[t] = 0.0
        else:
            new_shares[t] = float((total * targ[i]) / p)

    holdings2 = dict(holdings)
    holdings2["shares"] = new_shares
    holdings2["cash"] = 0.0
    return holdings2

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

def make_rebalance_table(df_alloc: pd.DataFrame, to="target"):
    total_mv = float(df_alloc["mkt_value"].sum())
    if total_mv <= 0:
        out = df_alloc[["ticker","price","mkt_value","weight","target_w","shares"]].copy()
        out["rebal_target"] = df_alloc["target_w"].to_numpy(float)
        out["$delta"] = 0.0
        return out

    if to == "band_mid":
        rebal_target = []
        for _, r in df_alloc.iterrows():
            rebal_target.append(0.5*(float(r["band_lo"])+float(r["band_hi"])))
        targ = np.array(rebal_target, float)
    else:
        targ = df_alloc["target_w"].to_numpy(float)

    delta = (targ - df_alloc["weight"].to_numpy(float)) * total_mv
    out = df_alloc[["ticker","price","shares","mkt_value","weight","target_w","band_lo","band_hi","status"]].copy()
    out["rebal_target"] = targ
    out["$delta"] = delta
    return out.sort_values("$delta")

# ============================ Email ============================

def format_email_table(df, title="Current Allocation"):
    lines = [title, "-"*len(title)]
    lines.append(f"{'Ticker':<10}{'Weight':>8}  {'Target':>8}  {'Band':>14}  {'Status':>7}")
    for _, r in df.iterrows():
        band = f"[{float(r.band_lo)*100:.1f}–{float(r.band_hi)*100:.1f}%]"
        lines.append(f"{r.ticker:<10}{float(r.weight)*100:>7.2f}%  {float(r.target_w)*100:>7.1f}%  {band:>14}  {r.status:>7}")
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
            buf = io.StringIO()
            df.to_csv(buf, index=False)
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

# ============================ History ============================

def update_history(history_path, dt_label, mv_tr, roc_tr):
    """
    Upsert one row per calendar date.
    """
    path = Path(history_path)
    cols = ["date", "mv_tr", "roc_total_return"]
    day = pd.to_datetime(str(dt_label)).date().isoformat()
    new_row = pd.DataFrame([{
        "date": day,
        "mv_tr": float(mv_tr),
        "roc_total_return": float(roc_tr),
    }], columns=cols)

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
    df_out = df.copy()
    df_out["date"] = pd.to_datetime(df_out["date"]).dt.strftime("%Y-%m-%d")
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


# ============================ Jobs ============================

def daily_job():
    try:
        port = load_portfolio()
        tickers = port["ticker"].tolist()

        # prices
        data = _yf_download(tickers, period="10d", interval="1d", tries=3)
        last_close = _last_series(data, "Close", tickers)
        last_adj   = _last_series(data, "Adj Close", tickers)
        if isinstance(data.index, pd.DatetimeIndex) and len(data.index) >= 1:
            last_ts = pd.Timestamp(data.index[-1]).tz_localize(None)
        else:
            last_ts = pd.Timestamp.utcnow().tz_localize(None)
        as_of = _today_from_ts(last_ts)

        # baseline + holdings + rebalance state
        baseline = load_or_create_baseline(tickers, baseline_date=BASELINE_DATE)
        holdings = load_or_init_holdings(port, baseline)
        rstate   = load_rebalance_state()
        rstate["lockup_days"] = int(rstate.get("lockup_days", LOCKUP_DAYS))  # keep synced

        # allocation from holdings
        alloc, total_mv_price = compute_allocation_from_holdings(port, last_close, holdings)
        signals = detect_signals(alloc)

        # lockup status
        locked, remaining_days = in_lockup(as_of, rstate)

        # signal-state change detection (pre-rebalance)
        prev_state = load_signals_state()
        today_state = { s["ticker"]: s["status"] for s in signals }
        changed = (today_state != prev_state)

        auto_rebalanced = False
        suppressed = False

        # AUTO REBALANCE LOGIC
        if signals and AUTO_REBALANCE_ON_BREACH:
            if locked:
                # match optimizer behavior: suppress rebalance during lockup
                rstate["suppressed_signals"] = int(rstate.get("suppressed_signals", 0)) + 1
                suppressed = True
            else:
                # execute synthetic rebalance at EOD prices
                holdings2 = rebalance_holdings_to_target(port, last_close, holdings)
                holdings2["as_of"] = as_of.isoformat()
                save_holdings(holdings2)
                holdings = holdings2

                set_lockup(as_of, rstate)
                save_rebalance_state(rstate)

                # recompute allocation after rebalance (should be on-target)
                alloc, total_mv_price = compute_allocation_from_holdings(port, last_close, holdings)
                signals = detect_signals(alloc)  # should clear unless band is narrower than exact target
                today_state = { s["ticker"]: s["status"] for s in signals }
                changed = (today_state != prev_state)

                auto_rebalanced = True

        # persist signal state (post-rebalance if it happened)
        save_signals_state(today_state)

        # simple TR proxy MV using Adj Close (holdings-based)
        # compute MV today + yesterday for roc
        d2 = _yf_download(tickers, period="3d", interval="1d", tries=3)
        a_prev, a_now = _two_series(d2, "Adj Close", tickers)

        shares = pd.Series(holdings.get("shares", {}), dtype="float64").reindex(tickers).fillna(0.0)
        mv_prev_tr = float(np.nansum(shares.to_numpy(float) * a_prev.reindex(tickers).to_numpy(float)))
        mv_now_tr  = float(np.nansum(shares.to_numpy(float) * a_now.reindex(tickers).to_numpy(float)))
        roc_tr = (mv_now_tr / mv_prev_tr - 1.0) if mv_prev_tr > 0 else 0.0

        update_history(HISTORY_CSV, as_of.isoformat(), mv_now_tr, roc_tr)

        # status snapshot
        snapshot = {
            "as_of": as_of.isoformat(),
            "total_mv_price": float(total_mv_price),
            "total_mv_tr": float(mv_now_tr),
            "roc_total_return": float(roc_tr),
            "auto_rebalance_on_breach": bool(AUTO_REBALANCE_ON_BREACH),
            "last_rebalance_date": rstate.get("last_rebalance_date"),
            "locked_until": rstate.get("locked_until"),
            "lockup_days": int(rstate.get("lockup_days", LOCKUP_DAYS)),
            "suppressed_signals": int(rstate.get("suppressed_signals", 0)),
            "rebalances": int(rstate.get("rebalances", 0)),
            "allocation": alloc[["ticker","price","shares","mkt_value","weight","target_w","band_lo","band_hi","status"]].to_dict(orient="records"),
        }
        Path(STATUS_JSON).write_text(json.dumps(snapshot, indent=2))
        save_rebalance_state(rstate)

        # compose email decision
        should_email = False
        if signals:
            should_email = ALWAYS_ALERT_ON_BREACH or changed or auto_rebalanced
        else:
            # if you want "cleared" emails, we send when changed and previous had breaches
            if changed and prev_state:
                should_email = True

        if should_email:
            locked2, remaining_days2 = in_lockup(as_of, rstate)
            lockup_line = (
                f"Lockup: {int(rstate.get('lockup_days', LOCKUP_DAYS))} calendar days\n"
                f"Last rebalance: {rstate.get('last_rebalance_date')}\n"
                f"Locked until:  {rstate.get('locked_until')} "
                f"({'IN LOCKUP' if locked2 else 'not locked'}; remaining {remaining_days2}d)\n"
                f"Suppressed signals (lifetime): {int(rstate.get('suppressed_signals', 0))}\n"
                f"Rebalances executed (lifetime): {int(rstate.get('rebalances', 0))}\n"
            )

            header = f"As of {as_of.isoformat()} (EOD)"
            body  = header + "\n\n" + lockup_line + "\n"
            body += format_email_table(alloc, "Current Allocation") + "\n\n"

            if auto_rebalanced:
                body += "✅ AUTO REBALANCE EXECUTED (synthetic holdings updated)\n\n"
            if suppressed:
                body += "⛔ Rebalance SUPPRESSED due to lockup.\n\n"

            if signals:
                body += "⚠️ Signals (band breaches):\n"
                for s in signals:
                    body += (
                        f"- {s['ticker']}: {s['status']} "
                        f"(w={s['weight']*100:.2f}% vs band {s['lo']*100:.1f}–{s['hi']*100:.1f}%, "
                        f"target {s['target']*100:.1f}%)\n"
                    )
            else:
                if prev_state:
                    body += "✅ Signals cleared.\n"

            reb = make_rebalance_table(alloc, to="target")
            email_send(
                subject=("Portfolio — AUTO REBALANCED" if auto_rebalanced else
                         "Portfolio — SIGNAL alert" if signals else
                         "Portfolio — Signals cleared"),
                body=body,
                attachments=[
                    ("allocation.csv", alloc[["ticker","price","shares","mkt_value","weight","target_w","band_lo","band_hi","status"]]),
                    ("rebalance_to_target.csv", reb),
                ]
            )
        else:
            print("No email sent (no breaches or unchanged).", flush=True)

    except Exception as e:
        _err("daily_job crashed:", repr(e))
        raise

def weekly_job():
    try:
        port = load_portfolio()
        tickers = port["ticker"].tolist()

        data = _yf_download(tickers, period="10d", interval="1d", tries=3)
        last_close = _last_series(data, "Close", tickers)
        if isinstance(data.index, pd.DatetimeIndex) and len(data.index) >= 1:
            last_ts = pd.Timestamp(data.index[-1]).tz_localize(None)
        else:
            last_ts = pd.Timestamp.utcnow().tz_localize(None)
        as_of = _today_from_ts(last_ts)

        baseline = load_or_create_baseline(tickers, baseline_date=BASELINE_DATE)
        holdings = load_or_init_holdings(port, baseline)
        rstate   = load_rebalance_state()

        alloc, total_mv = compute_allocation_from_holdings(port, last_close, holdings)

        hist = _load_history_collapsed(HISTORY_CSV)

        def span_return(colname, days):
            if hist is None or len(hist) < 2:
                return np.nan
            end = float(hist[colname].iloc[-1])
            start_idx = max(0, len(hist)-1-days)
            start = float(hist[colname].iloc[start_idx])
            return (end / start - 1.0) if start else np.nan

        ret_1w = span_return("mv_tr", 5)

        ret_mtd = np.nan
        if hist is not None and len(hist):
            this_month = hist[hist["date"].dt.to_period("M") == hist["date"].iloc[-1].to_period("M")]
            if len(this_month) >= 2:
                ret_mtd = float(this_month["mv_tr"].iloc[-1] / this_month["mv_tr"].iloc[0] - 1.0)

        ret_ytd = np.nan
        if hist is not None and len(hist):
            this_year = hist[hist["date"].dt.year == hist["date"].iloc[-1].year]
            if len(this_year) >= 2:
                ret_ytd = float(this_year["mv_tr"].iloc[-1] / this_year["mv_tr"].iloc[0] - 1.0)

        locked, remaining_days = in_lockup(as_of, rstate)
        lockup_line = (
            f"Lockup: {int(rstate.get('lockup_days', LOCKUP_DAYS))} calendar days\n"
            f"Last rebalance: {rstate.get('last_rebalance_date')}\n"
            f"Locked until:  {rstate.get('locked_until')} "
            f"({'IN LOCKUP' if locked else 'not locked'}; remaining {remaining_days}d)\n"
            f"Suppressed signals (lifetime): {int(rstate.get('suppressed_signals', 0))}\n"
            f"Rebalances executed (lifetime): {int(rstate.get('rebalances', 0))}\n"
        )

        header = f"Weekly recap — as of {as_of.isoformat()} (baseline {baseline.get('as_of')})"
        body  = header + "\n\n" + lockup_line + "\n"
        body += format_email_table(alloc, "Current Allocation") + "\n\n"
        body += "Performance (Adj Close proxy, holdings-based):\n"
        body += (f"- 1w:  {ret_1w*100:6.2f}%\n" if np.isfinite(ret_1w) else "- 1w:   n/a\n")
        body += (f"- MTD: {ret_mtd*100:6.2f}%\n" if np.isfinite(ret_mtd) else "- MTD:  n/a\n")
        body += (f"- YTD: {ret_ytd*100:6.2f}%\n" if np.isfinite(ret_ytd) else "- YTD:  n/a\n")

        reb_target = make_rebalance_table(alloc, to="target")
        reb_mid    = make_rebalance_table(alloc, to="band_mid")

        email_send(
            subject="Portfolio — Weekly recap",
            body=body,
            attachments=[
                ("allocation.csv", alloc[["ticker","price","shares","mkt_value","weight","target_w","band_lo","band_hi","status"]]),
                ("rebalance_to_target.csv", reb_target),
                ("rebalance_to_band_mid.csv", reb_mid),
                ("history.csv", pd.read_csv(HISTORY_CSV) if Path(HISTORY_CSV).exists() else pd.DataFrame()),
            ]
        )

    except Exception as e:
        _err("weekly_job crashed:", repr(e))
        raise

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
