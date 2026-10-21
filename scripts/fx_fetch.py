"""Download Dukascopy bid/ask candles (M1 and H1) into one Parquet file per symbol, timeframe and year.

Runs inside the GitHub Actions workflow .github/workflows/fx-data.yml.
Needs only pandas + pyarrow; the
candle files (.bi5, LZMA) come straight from datafeed.dukascopy.com:
- M1: one file per day and side,
- H1: one file per month and side (the running month is built from M1 days).

Dukascopy throttles GitHub runners to roughly 0.06 files/s per job (~0.25/s
for 4 jobs), so a full M1 history (~200k files without Saturdays) takes many
runner-hours. The workflow therefore works through
a plan of (symbol, timeframe, year) items in time-boxed jobs and is re-run on
a schedule until everything is there: H1 first, then M1 from the newest year
backwards.

    python3 scripts/fx_fetch.py --plan --assets assets.tsv --jobs 20    # matrix for the workflow
    python3 scripts/fx_fetch.py --items EURUSD:h1:2025,EURUSD:m1:2024 --upload fx-data
    python3 scripts/fx_fetch.py --probe | --rate-test                     # diagnostics

Asset names: <SYMBOL>_<YEAR>_<tf>.parquet with UTC bars and the columns
time, bid_open..bid_close, ask_open..ask_close, volume (bid side).
"""
import argparse
import datetime as dt
import json
import lzma
import os
import struct
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# symbol: (first day with minute candles, price divisor, plausible price range)
INSTRUMENTS = {
    "EURUSD": ("2003-05-04", 1e5, (0.8, 1.7)),
    "GBPUSD": ("2003-05-04", 1e5, (1.0, 2.2)),
    "AUDUSD": ("2003-08-03", 1e5, (0.4, 1.2)),
    "NZDUSD": ("2003-08-03", 1e5, (0.4, 1.0)),
    "USDCAD": ("2003-08-03", 1e5, (0.9, 1.7)),
    "USDCHF": ("2003-05-04", 1e5, (0.6, 1.8)),
    "USDJPY": ("2003-05-04", 1e3, (70, 220)),
    "EURJPY": ("2003-08-03", 1e3, (90, 250)),
    "GBPJPY": ("2003-08-03", 1e3, (110, 300)),
    "AUDJPY": ("2003-11-30", 1e3, (50, 150)),
    "EURGBP": ("2003-08-03", 1e5, (0.6, 1.0)),
    "EURCHF": ("2003-08-03", 1e5, (0.7, 1.7)),
    "XAUUSD": ("2003-05-05", 1e3, (250, 12000)),
    "XAGUSD": ("2003-05-04", 1e3, (3, 250)),
}
SYMBOLS = list(INSTRUMENTS)
TFS = ("h1", "m1")

ROOT = "https://datafeed.dukascopy.com/datafeed/{sym}/{y:04d}/{m:02d}"
URL_M1 = ROOT + "/{d:02d}/{side}_candles_min_1.bi5"
URL_H1 = ROOT + "/{side}_candles_hour_1.bi5"
OHLC = ["open", "high", "low", "close"]
COLUMNS = ["time"] + [f"{s}_{c}" for s in ("bid", "ask") for c in OHLC] + ["volume"]
RECORD = struct.Struct(">IIIIIf")  # seconds since file start, open, close, low, high, volume
WORKERS = 4
RETRY_STATUS = {202, 403, 429, 500, 502, 503, 504}
STATS = {"requests": 0, "retries": 0}

# Planning: estimated minutes per item at the measured ~0.06 files/s per job
# (4 jobs, 2026-10-21), job budget and refresh age.
JOB_RATE = 0.06
JOB_BUDGET_MIN = 300
REFRESH_DAYS = 7


def first_minute(symbol):
    return dt.date.fromisoformat(INSTRUMENTS[symbol][0])


def asset_name(symbol, year, tf):
    return f"{symbol}_{year}_{tf}.parquet"


# ---------------------------------------------------------------- download + decode

def fetch_bytes(url, patience_s=None):
    """GET with backoff for up to `patience_s` seconds (Dukascopy throttles with
    503 for minutes at a time; FX_PATIENCE, default 10 min). Returns b"" for
    periods without data (404); raises when the patience is used up."""
    delay = 2.0
    patience_s = float(os.environ.get("FX_PATIENCE", 600)) if patience_s is None else patience_s
    deadline = time.time() + patience_s
    attempt = 0
    while True:
        STATS["requests"] += 1
        if attempt:
            STATS["retries"] += 1
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                if r.status == 200:
                    return r.read()
                status = r.status
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return b""
            status = e.code
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            status = repr(e)
        if time.time() + delay > deadline or (isinstance(status, int) and status not in RETRY_STATUS):
            raise RuntimeError(f"{url}: {status} after {attempt + 1} attempts")
        time.sleep(delay)
        delay = min(delay * 2, 300)
        attempt += 1


def decode(raw, start, divisor):
    """Decode one .bi5 candle file whose offsets count seconds from `start` (UTC)."""
    if not raw:
        return None
    data = lzma.decompress(raw)
    n = len(data) // RECORD.size
    arr = np.frombuffer(data[: n * RECORD.size], dtype=">u4").reshape(n, 6)
    vol = np.frombuffer(data[: n * RECORD.size], dtype=">f4").reshape(n, 6)[:, 5]
    t0 = pd.Timestamp(start, tz="UTC")
    df = pd.DataFrame({
        "time": t0 + pd.to_timedelta(arr[:, 0].astype("int64"), unit="s"),
        "open": arr[:, 1] / divisor, "close": arr[:, 2] / divisor,
        "low": arr[:, 3] / divisor, "high": arr[:, 4] / divisor,
        "volume": vol.astype("float64"),
    })
    # Dukascopy pads non-trading periods with flat zero-volume candles.
    return df[df["volume"] > 0]


decode_day = decode  # backwards-compatible name


def _try_fetch(url):
    try:
        return fetch_bytes(url)
    except RuntimeError as e:
        print(f"skip {e}", flush=True)
        return None


def _fetch_many(symbol, side, jobs):
    """jobs: list of (url, start). Returns (candles, starts whose file could not be fetched)."""
    divisor = INSTRUMENTS[symbol][1]
    with ThreadPoolExecutor(int(os.environ.get("FX_WORKERS", WORKERS))) as ex:
        raws = list(ex.map(_try_fetch, [u for u, _ in jobs]))
    failed = [s for r, (_, s) in zip(raws, jobs) if r is None]
    parts = [decode(r, s, divisor) for r, (_, s) in zip(raws, jobs) if r is not None]
    parts = [p for p in parts if p is not None and len(p)]
    if not parts:
        return pd.DataFrame(columns=["time"] + OHLC + ["volume"]), failed
    return pd.concat(parts, ignore_index=True), failed


def fetch_side(symbol, side, days):
    """Minute candles for a list of days."""
    jobs = [(URL_M1.format(sym=symbol, y=d.year, m=d.month - 1, d=d.day, side=side.upper()), d) for d in days]
    return _fetch_many(symbol, side, jobs)


def fetch_side_hours(symbol, side, months):
    """Hour candles for a list of month starts."""
    jobs = [(URL_H1.format(sym=symbol, y=m.year, m=m.month - 1, side=side.upper()), m) for m in months]
    return _fetch_many(symbol, side, jobs)


def minutes_to_hours(df):
    """Aggregate one side's minute candles to hour candles."""
    if df.empty:
        return df
    r = df.set_index("time").resample("1h")
    out = pd.DataFrame({"open": r["open"].first(), "high": r["high"].max(), "low": r["low"].min(),
                        "close": r["close"].last(), "volume": r["volume"].sum()})
    return out.dropna(subset=["open"]).reset_index()


def merge_sides(bid, ask):
    """Join bid and ask candles on their time."""
    b = bid.rename(columns={c: f"bid_{c}" for c in OHLC})
    a = ask.rename(columns={c: f"ask_{c}" for c in OHLC}).drop(columns=["volume"], errors="ignore")
    df = b.merge(a, on="time", how="inner")
    df = df.drop_duplicates("time").sort_values("time")
    df["time"] = df["time"].dt.as_unit("ms")
    for c in COLUMNS[1:]:
        df[c] = df[c].astype("float64")
    return df[COLUMNS].reset_index(drop=True)


def check(df, symbol):
    """Raise if prices are outside the plausible range; report crossed quotes."""
    lo, hi = INSTRUMENTS[symbol][2]
    mid = (df["bid_close"] + df["ask_close"]) / 2
    if not mid.between(lo, hi).all():
        raise ValueError(f"{symbol}: prices outside {lo}..{hi}: {mid.min()} .. {mid.max()}")
    crossed = (df["ask_close"] < df["bid_close"]).mean()
    if crossed > 0.01:
        # Early Dukascopy history (e.g. JPY pairs 2004) has crossed quotes; keep and report.
        print(f"WARN {symbol}: ask < bid in {crossed:.1%} of bars", flush=True)


def write_parquet(df, path):
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, path, compression="zstd", compression_level=9,
                   use_byte_stream_split=COLUMNS[1:])


def days_between(start, end):
    """Days in [start, end) without Saturdays (UTC): FX and metals do not trade then
    and Dukascopy's Saturday files hold no candles."""
    days = (start + dt.timedelta(days=i) for i in range((end - start).days))
    return [d for d in days if d.weekday() != 5]


def fetch_days(symbol, days):
    """M1 bid/ask for the given days. Returns (bars, days that could not be fetched)."""
    bid, fb = fetch_side(symbol, "bid", days)
    ask, fa = fetch_side(symbol, "ask", days)
    failed = sorted(set(fb) | set(fa))
    df = merge_sides(bid, ask)
    if failed:  # keep only complete days (a day with one side missing is dropped)
        bad = {pd.Timestamp(d) for d in failed}
        df = df[~df["time"].dt.tz_convert(None).dt.normalize().isin(bad)].reset_index(drop=True)
    return df, failed


def fetch_range(symbol, start, end):
    """M1 bid/ask for [start, end); raises if any day could not be fetched."""
    df, failed = fetch_days(symbol, days_between(start, end))
    if failed:
        raise RuntimeError(f"{symbol}: {len(failed)} days not fetched, first {failed[0]}")
    return df


def fetch_hours(symbol, start, end, today):
    """H1 bid/ask for [start, end): monthly files, the running month from M1 days."""
    month0 = dt.date(today.year, today.month, 1)
    months, m = [], dt.date(start.year, start.month, 1)
    while m < end and m < month0:
        months.append(m)
        m = dt.date(m.year + (m.month == 12), m.month % 12 + 1, 1)
    sides = {}
    for side in ("bid", "ask"):
        parts, failed = [], []
        if months:
            df, f = fetch_side_hours(symbol, side, months)
            parts.append(df)
            failed += f
        if end > month0:
            df, f = fetch_side(symbol, side, days_between(max(start, month0), end))
            parts.append(minutes_to_hours(df))
            failed += f
        if failed:
            raise RuntimeError(f"{symbol} h1 {side}: {len(failed)} files not fetched, first {failed[0]}")
        parts = [p for p in parts if len(p)]
        sides[side] = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["time"] + OHLC + ["volume"])
    df = merge_sides(sides["bid"], sides["ask"])
    t0, t1 = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
    return df[(df.time >= t0) & (df.time < t1)].reset_index(drop=True)


MAX_MISSING_SHARE = 0.25  # more failed days than this: keep nothing, retry the whole year


def missing_name(symbol, year, tf):
    return f"{symbol}_{year}_{tf}.missing.txt"


def _release_download(tag, name, out_dir):
    subprocess.run(["gh", "release", "download", tag, "-p", name, "-D", str(out_dir), "--clobber"], check=True)
    return Path(out_dir) / name


def fetch_item(symbol, tf, year, today, out_dir, repair_from=None):
    """Download one (symbol, tf, year). Returns the files to upload.

    M1 tolerates single days that stay unreachable: the year is stored without
    them and <name>.missing.txt lists them. With repair_from (a release tag)
    the existing file and list are downloaded and only those days fetched.
    """
    start = max(dt.date(year, 1, 1), first_minute(symbol))
    end = min(dt.date(year + 1, 1, 1), today)  # today's data is not complete yet
    if start >= end:
        print(f"{symbol} {year} {tf}: outside the available range", flush=True)
        return []
    t = time.time()
    out_dir = Path(out_dir)
    path = out_dir / asset_name(symbol, year, tf)
    miss = out_dir / missing_name(symbol, year, tf)
    if tf != "m1":
        df, failed = fetch_hours(symbol, start, end, today), []
    elif repair_from:
        old = pd.read_parquet(_release_download(repair_from, path.name, out_dir))
        todo = [dt.date.fromisoformat(x) for x in
                _release_download(repair_from, miss.name, out_dir).read_text().split()]
        new, failed = fetch_days(symbol, todo)
        df = pd.concat([old, new], ignore_index=True).drop_duplicates("time").sort_values("time")
        df = df.reset_index(drop=True)
        print(f"{symbol} {year} m1 repair: {len(todo) - len(failed)} of {len(todo)} days fetched", flush=True)
    else:
        days = days_between(start, end)
        df, failed = fetch_days(symbol, days)
        if len(failed) > MAX_MISSING_SHARE * len(days):
            raise RuntimeError(f"{symbol} {year} m1: {len(failed)} of {len(days)} days not fetched")
    if df.empty:
        print(f"{symbol} {year} {tf}: no data", flush=True)
        return []
    check(df, symbol)
    write_parquet(df, path)
    spread = (df["ask_close"] - df["bid_close"]).median()
    print(f"{symbol} {year} {tf}: {len(df)} bars {df.time.iloc[0]} .. {df.time.iloc[-1]} "
          f"median spread {spread:.6g} -> {path.stat().st_size / 1e6:.1f} MB "
          f"in {time.time() - t:.0f}s" + (f", {len(failed)} days still missing" if failed else ""), flush=True)
    files = [path]
    if failed:
        miss.write_text("\n".join(d.isoformat() for d in failed) + "\n")
        files.append(miss)
    elif repair_from:
        subprocess.run(["gh", "release", "delete-asset", repair_from, miss.name, "-y"], check=False)
    return files


# ---------------------------------------------------------------- planning

def estimate_minutes(symbol, tf, year, today):
    if tf.endswith("r"):
        return 15  # repair: a handful of days
    start = max(dt.date(year, 1, 1), first_minute(symbol))
    end = min(dt.date(year + 1, 1, 1), today)
    days = len(days_between(start, end)) if end > start else 0
    if tf == "m1":
        files = 2 * days
    else:
        files = 2 * (days / 30.4 + (today.day if year == today.year else 0))
    return 3 + files / JOB_RATE / 60


def plan(assets, today, symbols=None, tfs=TFS, full=False, years=None):
    """Items still to do, in priority order: H1 before M1, newest years first.

    assets: {name: updated_at (datetime)}. A current-year file older than
    REFRESH_DAYS is refreshed; a previous-year file written before that year
    ended is refreshed once.
    """
    items = []
    for tf in ("h1", "m1"):
        if tf not in tfs:
            continue
        for year in range(today.year, 2002, -1):
            if years and year not in years:
                continue
            for sym in symbols or SYMBOLS:
                if year < first_minute(sym).year:
                    continue
                upd = assets.get(asset_name(sym, year, tf))
                year_end = dt.datetime(year + 1, 1, 1, tzinfo=dt.timezone.utc)
                if missing_name(sym, year, tf) in assets and upd is not None and not full:
                    items.append((sym, tf + "r", year))  # repair the listed days only
                    continue
                if full or upd is None:
                    todo = True
                elif year == today.year:
                    todo = (dt.datetime.now(dt.timezone.utc) - upd).days >= REFRESH_DAYS
                else:
                    todo = upd < year_end
                if todo:
                    items.append((sym, tf, year))
    return items


def pack(items, jobs, today, budget=JOB_BUDGET_MIN):
    """Greedy: fill up to `jobs` buckets of `budget` minutes in priority order."""
    buckets = [[] for _ in range(jobs)]
    load = [0.0] * jobs
    for it in items:
        cost = estimate_minutes(*it, today)
        k = min(range(jobs), key=lambda i: load[i])
        if load[k] + cost > budget and buckets[k]:
            continue  # does not fit this run; a later run picks it up
        buckets[k].append(it)
        load[k] += cost
    return [b for b in buckets if b]


MIN_ASSET_BYTES = 20_000  # a full H1 year is ~250 KB, M1 ~13 MB; smaller = broken download


def read_assets(path):
    """Read 'name<TAB>updated_at[<TAB>size]' lines (from gh api).

    Suspiciously small files of past years are left out, so the planner
    downloads them again.
    """
    assets = {}
    if path and Path(path).exists():
        for line in Path(path).read_text().splitlines():
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            name, upd = parts[0], parts[1]
            if len(parts) > 2 and int(parts[2]) < MIN_ASSET_BYTES and name.endswith(".parquet"):
                print(f"{name}: only {parts[2]} bytes, planned again", file=sys.stderr)
                continue
            assets[name] = dt.datetime.fromisoformat(upd.replace("Z", "+00:00"))
    return assets


def parse_years(s):
    if not s:
        return None
    if "-" in s:
        a, b = map(int, s.split("-"))
        return list(range(a, b + 1))
    return [int(y) for y in s.split(",")]


# ---------------------------------------------------------------- diagnostics

def rate_test():
    """Fetch four weeks of EURUSD M1 and report throughput (for tuning FX_WORKERS)."""
    t = time.time()
    df = fetch_range("EURUSD", dt.date(2023, 3, 6), dt.date(2023, 4, 3))
    sec = time.time() - t
    files = 2 * 28
    print(f"RATE workers={os.environ.get('FX_WORKERS', WORKERS)}: {files} files in {sec:.0f}s "
          f"= {files / sec:.2f} files/s, requests {STATS['requests']}, retries {STATS['retries']}, "
          f"rows {len(df)}", flush=True)


def probe():
    """Fetch one recent and one old week per symbol and print sanity checks."""
    ok = True
    for sym in SYMBOLS:
        for start in (dt.date(2025, 6, 2), max(first_minute(sym) + dt.timedelta(days=60), dt.date(2004, 1, 5))):
            try:
                df = fetch_range(sym, start, start + dt.timedelta(days=7))
                check(df, sym)
                spread = (df["ask_close"] - df["bid_close"]).median()
                print(f"OK   {sym} {start}: {len(df)} rows, close {df.bid_close.iloc[-1]:.5g}, "
                      f"median spread {spread:.3g}", flush=True)
            except Exception as e:  # report every symbol, fail at the end
                ok = False
                print(f"FAIL {sym} {start}: {e}", flush=True)
    if not ok:
        sys.exit(1)


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true", help="print the job matrix as GitHub output")
    ap.add_argument("--assets", help="file with 'name<TAB>updated_at' of existing release assets")
    ap.add_argument("--jobs", type=int, default=20)
    ap.add_argument("--symbols", default="", help="comma-separated, empty = all")
    ap.add_argument("--tfs", default="h1,m1")
    ap.add_argument("--years", default="", help="e.g. 2025 or 2003-2010")
    ap.add_argument("--full", action="store_true", help="re-download everything selected")
    ap.add_argument("--items", help="SYMBOL:tf:YEAR,... to download")
    ap.add_argument("--budget-min", type=float, default=330, help="stop starting new items after this")
    ap.add_argument("--out", default="out")
    ap.add_argument("--upload", metavar="TAG", help="upload each file to this release with gh")
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--rate-test", action="store_true")
    args = ap.parse_args()
    today = dt.datetime.now(dt.timezone.utc).date()

    if args.probe:
        return probe()
    if args.rate_test:
        return rate_test()

    if args.plan:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()] or SYMBOLS
        unknown = sorted(set(symbols) - set(SYMBOLS))
        if unknown:
            sys.exit(f"unknown symbols: {unknown}")
        tfs = [t.strip() for t in args.tfs.split(",") if t.strip()]
        items = plan(read_assets(args.assets), today, symbols, tfs, args.full, parse_years(args.years))
        buckets = pack(items, args.jobs, today)
        matrix = [{"id": i, "items": ",".join(f"{s}:{t}:{y}" for s, t, y in b)} for i, b in enumerate(buckets)]
        print(f"Plan: {len(items)} offene Einträge, {sum(len(b) for b in buckets)} in diesem Lauf "
              f"auf {len(buckets)} Jobs", file=sys.stderr)
        print(f"matrix={json.dumps(matrix)}")
        print(f"count={len(matrix)}")
        return

    if not args.items:
        sys.exit("nothing to do: pass --items, --plan, --probe or --rate-test")
    Path(args.out).mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    failed = []
    for it in args.items.split(","):
        sym, tf, year = it.split(":")
        if (time.time() - t0) / 60 + estimate_minutes(sym, tf, int(year), today) > args.budget_min:
            print(f"budget reached, leaving {it} and the rest for the next run", flush=True)
            break
        repair = tf.endswith("r")
        try:
            files = fetch_item(sym, tf.rstrip("r"), int(year), today, args.out,
                               repair_from=args.upload if repair else None)
        except Exception as e:  # keep going; the next scheduled run retries it
            print(f"FAIL {it}: {e}", flush=True)
            failed.append(it)
            continue
        if args.upload:
            for path in files:
                subprocess.run(["gh", "release", "upload", args.upload, str(path), "--clobber"], check=True)
                path.unlink()
    if failed:
        # Not fatal: the planner of the next scheduled run picks these up again.
        print(f"::warning::{len(failed)} Einträge fehlgeschlagen, nächster Lauf versucht sie erneut: "
              f"{','.join(failed)}", flush=True)


if __name__ == "__main__":
    main()
