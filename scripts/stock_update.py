"""Incremental update of the stock-data release (Alpaca Market Data, SIP, adjustment=all).

Reads the current release files from SRC, fetches everything after the
manifest's data_until up to the last complete trading day, and writes the
changed files (plus manifest.json) to OUT:

    1day-<YEAR>.parquet      daily bars, all symbols of the latest year file
    1min-<SYM>.parquet       1-minute bars incl. pre/post market
    5min.parquet, 30min.parquet   regular trading hours only
    news-<YEAR>.parquet      Benzinga headlines, windows 04:00-09:30 and 16:00-19:00 New York
    earnings-events.parquet  results parsed from the headlines (see earnings.py)

Prices are split- and dividend-adjusted. When Alpaca re-adjusts history (split,
dividend), the overlap days no longer match; the stored history of that symbol
is then rescaled by the measured price (and volume) ratio so the series stays
continuous. The universe files are yearly and not touched here.

    APCA_API_KEY_ID=... APCA_API_SECRET_KEY=... \\
    python3 scripts/stock_update.py SRC OUT [--until YYYY-MM-DD]
"""
import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import earnings  # noqa: E402

BARS_URL = "https://data.alpaca.markets/v2/stocks/bars"
NEWS_URL = "https://data.alpaca.markets/v1beta1/news"
NY = ZoneInfo("America/New_York")
PRICES = ["open", "high", "low", "close"]
COLS = ["symbol", "time"] + PRICES + ["volume"]
OVERLAP_DAYS = 7          # re-fetched days before data_until, used to detect re-adjustment
EARNINGS_CONTEXT = 45     # days of headlines used to rebuild recent earnings events
EARNINGS_REPLACE = 10     # events with d later than data_until - this are rebuilt


# ---------------------------------------------------------------- Alpaca

class Throttle:
    """At most `per_min` requests per minute (Alpaca allows 200)."""

    def __init__(self, per_min=180):
        self.gap, self.last = 60 / per_min, 0.0

    def wait(self):
        dt = time.monotonic() - self.last
        if dt < self.gap:
            time.sleep(self.gap - dt)
        self.last = time.monotonic()


THROTTLE = Throttle()


def _get(url, params):
    headers = {"APCA-API-KEY-ID": os.environ["APCA_API_KEY_ID"],
               "APCA-API-SECRET-KEY": os.environ["APCA_API_SECRET_KEY"]}
    full = url + "?" + urllib.parse.urlencode(params)
    for attempt in range(8):
        THROTTLE.wait()
        try:
            with urllib.request.urlopen(urllib.request.Request(full, headers=headers), timeout=60) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code != 429 and e.code < 500:
                raise RuntimeError(f"{url}: {e.code} {e.read()[:200]!r}") from e
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            pass
        time.sleep(min(60, 2 ** attempt))
    raise RuntimeError(f"request failed: {full}")


def fetch_bars(symbols, timeframe, start, end, chunk=100):
    """Bars for many symbols as one DataFrame in the bundle format (time UTC)."""
    rows = []
    for i in range(0, len(symbols), chunk):
        params = {"symbols": ",".join(symbols[i:i + chunk]), "timeframe": timeframe,
                  "start": start, "end": end, "adjustment": "all", "feed": "sip",
                  "limit": 10000, "sort": "asc"}
        while True:
            j = _get(BARS_URL, params)
            for sym, bars in (j.get("bars") or {}).items():
                for b in bars:
                    rows.append((sym, b["t"], b["o"], b["h"], b["l"], b["c"], b["v"]))
            token = j.get("next_page_token")
            if not token:
                break
            params["page_token"] = token
    df = pd.DataFrame(rows, columns=COLS)
    df["symbol"] = df["symbol"].astype("string")
    df["time"] = pd.to_datetime(df["time"], utc=True).dt.as_unit("us")
    df[PRICES] = df[PRICES].astype("float32")
    df["volume"] = df["volume"].astype("int64")
    return df


def fetch_news_day(d):
    out = []
    for lo, hi in (("04:00", "09:30"), ("16:00", "19:00")):
        params = {"start": datetime.fromisoformat(f"{d}T{lo}").replace(tzinfo=NY).isoformat(),
                  "end": datetime.fromisoformat(f"{d}T{hi}").replace(tzinfo=NY).isoformat(),
                  "limit": 50, "sort": "asc"}
        while True:
            j = _get(NEWS_URL, params)
            for n in j.get("news") or []:
                out.append({"day": d.isoformat(), "id": n["id"], "t": n["created_at"],
                            "headline": n["headline"], "symbols": ",".join(n["symbols"])})
            token = j.get("next_page_token")
            if not token:
                break
            params["page_token"] = token
    return out


# ---------------------------------------------------------------- merge logic

def ny_day(t):
    return t.dt.tz_convert(NY).dt.date


def adjust_and_append(old, new, until_old, until_new=None, regular_only=False, log=None):
    """Rescale `old` where Alpaca re-adjusted the overlap, then append days after until_old.

    old/new: bundle-format frames. until_old: last day (NY) already in `old`.
    until_new: last day (NY) to append; later (incomplete) days are dropped.
    Returns (frame, {symbol: (price_ratio, volume_ratio)}).
    """
    if regular_only and len(new):
        local = new["time"].dt.tz_convert(NY)
        hm = local.dt.hour * 60 + local.dt.minute
        new = new[(hm >= 9 * 60 + 30) & (hm < 16 * 60)]
    new_day = ny_day(new["time"])
    overlap = new[new_day <= until_old]
    fresh = new[(new_day > until_old) & ((new_day <= until_new) if until_new else True)]
    changes = {}
    if len(overlap):
        m = overlap.merge(old, on=["symbol", "time"], suffixes=("_n", "_o"))
        m = m[(m["close_o"] > 0) & (m["close_n"] > 0)]
        for sym, g in m.groupby("symbol"):
            r = float(np.median(g["close_n"].astype("float64") / g["close_o"].astype("float64")))
            v = g[g["volume_o"] > 0]
            vr = float(np.median(v["volume_n"] / v["volume_o"])) if len(v) else 1.0
            if abs(r - 1) > 2e-4 or abs(vr - 1) > 0.02:
                changes[sym] = (r, vr)
    if changes:
        old = old.copy()
        for sym, (r, vr) in changes.items():
            sel = old["symbol"] == sym
            for c in PRICES:
                old.loc[sel, c] = (old.loc[sel, c].astype("float64") * r).astype("float32")
            if abs(vr - 1) > 0.02:
                old.loc[sel, "volume"] = np.rint(old.loc[sel, "volume"] * vr).astype("int64")
        if log:
            log(f"  re-adjusted {len(changes)} symbols: " +
                ", ".join(f"{s} x{r:.4f}" for s, (r, _) in list(changes.items())[:10]) +
                (" ..." if len(changes) > 10 else ""))
    out = pd.concat([old, fresh], ignore_index=True)
    out = out.drop_duplicates(["symbol", "time"], keep="last").sort_values(["symbol", "time"])
    return out.reset_index(drop=True), changes


# ---------------------------------------------------------------- files

def describe(df):
    info = {"rows": len(df), "columns": list(df.columns)}
    if "symbol" in df:
        info["symbols"] = int(df["symbol"].nunique())
    for col in ("time", "t", "d"):
        if col in df and len(df):
            info["first"], info["last"] = str(df[col].min()), str(df[col].max())
            break
    return info


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class Writer:
    def __init__(self, out, manifest):
        self.out, self.manifest = Path(out), manifest
        self.out.mkdir(parents=True, exist_ok=True)

    def write(self, df, name):
        keys = [k for k in ("symbol", "time") if k in df]
        df = df.sort_values(keys) if keys else df
        path = self.out / name
        df.to_parquet(path, index=False, compression="zstd", compression_level=9)
        info = describe(df)
        info.update({"bytes": path.stat().st_size, "sha256": sha256(path)})
        self.manifest["files"][name] = info
        print(f"{name}: {len(df)} rows, {path.stat().st_size / 1e6:.1f} MB", flush=True)


def fix_na_symbol(df):
    """Ticker "NA" (Nano Labs) was read as missing value when the files were built."""
    if "symbol" in df and df["symbol"].isna().any():
        df = df.copy()
        df["symbol"] = df["symbol"].fillna("NA")
    return df


# ---------------------------------------------------------------- main

def last_complete_day(now=None):
    """Last NY trading-calendar day whose session (incl. post-market) is over."""
    now = (now or datetime.now(timezone.utc)).astimezone(NY)
    d = now.date() if now.hour >= 20 else now.date() - timedelta(days=1)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--until", help="last day to include (default: last complete day)")
    args = ap.parse_args()
    src = Path(args.src)
    manifest = json.loads((src / "manifest.json").read_text())
    since = date.fromisoformat(manifest["data_until"])
    until = date.fromisoformat(args.until) if args.until else last_complete_day()
    # Alpaca's free plan delays SIP data by 15 minutes
    end = min(datetime.combine(until + timedelta(days=1), datetime.min.time(), NY),
              datetime.now(timezone.utc) - timedelta(minutes=16)).isoformat()
    start = (since - timedelta(days=OVERLAP_DAYS)).isoformat()
    print(f"data_until {since} -> {until}", flush=True)
    if until <= since:
        print("nothing to do")
        return
    w = Writer(args.out, manifest)
    log = lambda s: print(s, flush=True)  # noqa: E731

    # daily bars: all symbols of the latest year file, across all year files
    years = sorted(int(p.stem.split("-")[1]) for p in src.glob("1day-*.parquet"))
    daily = pd.concat([pd.read_parquet(src / f"1day-{y}.parquet") for y in years], ignore_index=True)
    symbols = sorted(pd.read_parquet(src / f"1day-{years[-1]}.parquet", columns=["symbol"]).symbol.unique())
    log(f"1day: {len(symbols)} symbols")
    new = fetch_bars(symbols, "1Day", start, end)
    daily, changed = adjust_and_append(daily, new, since, until, log=log)
    trading_days = sorted(set(ny_day(daily.loc[daily.symbol == "SPY", "time"])))
    trading_days = [d for d in trading_days if d <= until]
    if trading_days[-1] <= since:
        print("no new trading day yet")
        return
    year_of = daily["time"].dt.tz_convert(NY).dt.year
    touched = {y for y in year_of[daily.symbol.isin(changed)].unique()} | set(range(since.year, until.year + 1))
    for y in sorted(touched):
        w.write(daily[year_of == y], f"1day-{y}.parquet")
    del daily, new

    # 1-minute bars, one file per symbol
    for p in sorted(src.glob("1min-*.parquet")):
        sym = p.stem.split("-", 1)[1]
        old = pd.read_parquet(p)
        new = fetch_bars([sym], "1Min", start, end)
        df, _ = adjust_and_append(old, new, since, until, log=log)
        w.write(df, p.name)

    # 5- and 30-minute bars, regular hours, all symbols in one file
    for tf, name in (("5Min", "5min.parquet"), ("30Min", "30min.parquet")):
        old = pd.read_parquet(src / name)
        syms = sorted(old.symbol.unique())
        log(f"{name}: {len(syms)} symbols")
        new = fetch_bars(syms, tf, start, end)
        df, _ = adjust_and_append(old, new, since, until, regular_only=True, log=log)
        w.write(df, name)

    # headlines
    days = [d for d in trading_days if d > since]
    items = []
    for d in days:
        items += fetch_news_day(d)
    log(f"news: {len(items)} headlines for {len(days)} days")
    fresh = pd.DataFrame(items, columns=["day", "id", "t", "headline", "symbols"])
    fresh["t"] = pd.to_datetime(fresh["t"], utc=True).dt.as_unit("us")
    for c in ("day", "headline", "symbols"):
        fresh[c] = fresh[c].astype("string")
    for y in sorted({d.year for d in days}):
        path = src / f"news-{y}.parquet"
        old = pd.read_parquet(path) if path.exists() else fresh.iloc[0:0]
        part = fresh[fresh["day"].str.startswith(str(y))]
        df = pd.concat([old, part], ignore_index=True).drop_duplicates("id").sort_values(["day", "t"])
        w.write(df.reset_index(drop=True), f"news-{y}.parquet")

    # earnings events: rebuild the recent ones from the last weeks of headlines
    lo = (since - timedelta(days=EARNINGS_CONTEXT)).isoformat()
    heads = []
    for y in range(int(lo[:4]), until.year + 1):
        f = Path(args.out) / f"news-{y}.parquet"
        f = f if f.exists() else src / f"news-{y}.parquet"
        if f.exists():
            heads.append(pd.read_parquet(f))
    h = pd.concat(heads, ignore_index=True)
    h = h[h["day"] >= lo]
    h = pd.DataFrame({"id": h["id"], "t": h["t"].dt.tz_convert(NY), "h": h["headline"].astype(str),
                      "s": h["symbols"].fillna("").map(lambda s: tuple(x for x in s.split(",") if x))})
    ev = earnings.build_events(h.sort_values("t").reset_index(drop=True), trading_days)
    cut = (since - timedelta(days=EARNINGS_REPLACE)).isoformat()
    old = fix_na_symbol(pd.read_parquet(src / "earnings-events.parquet"))
    if len(ev):
        ev["t"] = ev["t"].map(lambda t: t.isoformat(sep=" "))
        ev["d"] = ev["d"].map(lambda d: d.isoformat())
        ev = ev[ev["d"] > cut]
    keep = old[old["d"] <= cut]
    df = pd.concat([keep, ev[old.columns] if len(ev) else ev], ignore_index=True)
    for c in ("symbol", "t", "d", "when", "headline"):
        df[c] = df[c].astype("string")
    log(f"earnings: {len(old) - len(keep)} recent events replaced by {len(df) - len(keep)}")
    w.write(df.sort_values(["d", "symbol"]).reset_index(drop=True), "earnings-events.parquet")

    # one-off repair of the "NA" ticker
    stats = pd.read_parquet(src / "universe-yearly-stats.parquet")
    if stats["symbol"].isna().any():
        w.write(fix_na_symbol(stats), "universe-yearly-stats.parquet")

    manifest["data_until"] = trading_days[-1].isoformat()
    manifest["built_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    (Path(args.out) / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
    print(f"data_until now {manifest['data_until']}; {len(list(Path(args.out).iterdir()))} files written")


if __name__ == "__main__":
    main()
