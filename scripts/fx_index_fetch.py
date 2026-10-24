"""Raw data for RDT test 48 (intraday FX market index), fetched on GitHub runners.

Cloud sessions cannot reach the sources (their proxy answers 403), so this script
runs in .github/workflows/fx-index-data.yml and uploads every finished file to
the release `fx-index` of this repo. fx-index/request.json lists what to fetch;
a new run skips files that are already in the release.

    python3 scripts/fx_index_fetch.py plan --request fx-index/request.json --assets assets.tsv --jobs 3
    python3 scripts/fx_index_fetch.py probe --upload fx-index
    python3 scripts/fx_index_fetch.py misc --upload fx-index
    python3 scripts/fx_index_fetch.py duka --items t:EURUSD:2015-01-15:6-16,m1:DOLLARIDXUSD:2024-01
    python3 scripts/fx_index_fetch.py second --source fxcm --month 2024-08 --symbols EURUSD,USDJPY

Items of `duka` (Dukascopy datafeed, same throttling as scripts/fx_fetch.py):
    t:SYM:YYYY-MM-DD:H0-H1   ticks of the hours [H0, H1) UTC  -> duka_t_SYM_DATE_H0-H1.parquet
                             (time, bid, ask, bid_vol, ask_vol; one row per quote)
    m1:SYM:YYYY-MM           bid/ask minute candles of a month -> duka_m1_SYM_YYYY-MM.parquet
    h1:SYM:YYYY              bid/ask hour candles of a year    -> duka_h1_SYM_YYYY.parquet

`second` builds bid/ask minute bars from a second, independent tick source
(FXCM weekly files or Exness monthly archives) -> second_<src>_SYM_YYYY-MM.parquet
with the tick count per minute. `misc` stores the raw downloads of Yahoo (CME FX
futures and ICE DX futures), FRED (Fed dollar indices), BIS (Triennial Survey
turnover), the Fed FOMC calendars and the BLS release schedule.
"""
import argparse
import datetime as dt
import gzip
import io
import json
import lzma
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fx_fetch  # noqa: E402  (same folder; fetch_bytes, decode, merge_sides)

UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
DUKA_TICKS = "https://datafeed.dukascopy.com/datafeed/{sym}/{y:04d}/{m:02d}/{d:02d}/{h:02d}h_ticks.bi5"
TICK = np.dtype([("ms", ">u4"), ("ask", ">u4"), ("bid", ">u4"), ("ask_vol", ">f4"), ("bid_vol", ">f4")])
G8 = ["EUR", "GBP", "AUD", "NZD", "USD", "CAD", "CHF", "JPY"]  # market quoting priority
PAIRS = [a + b for i, a in enumerate(G8) for b in G8[i + 1:]]  # the 28 G8 pairs


def divisor(sym):
    """Dukascopy stores prices as integers: 3 decimals for JPY quotes and index CFDs."""
    return 1e3 if sym.endswith("JPY") or "IDX" in sym else 1e5


def http_get(url, timeout=60, tries=3, headers=None):
    """GET with a few retries; returns (status, bytes). 404 and other client errors are not retried."""
    status, body = None, b""
    for k in range(tries):
        req = urllib.request.Request(url, headers={"User-Agent": UA, **(headers or {})})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.status, r.read()
        except urllib.error.HTTPError as e:
            status, body = e.code, b""
            if e.code < 500 and e.code != 429:
                return status, body
        except Exception as e:  # network errors: retry
            status = repr(e)[:120]
        time.sleep(2 ** (k + 1))
    return status, body


def write(df, path):
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), path, compression="zstd", compression_level=9)


def upload(tag, paths):
    if not tag:
        return
    for p in paths:
        subprocess.run(["gh", "release", "upload", tag, str(p), "--clobber"], check=True)


# ---------------------------------------------------------------- Dukascopy

def decode_ticks(raw, hour_start, div):
    if not raw:
        return None
    data = lzma.decompress(raw)
    arr = np.frombuffer(data[: len(data) // TICK.itemsize * TICK.itemsize], dtype=TICK)
    t0 = int(pd.Timestamp(hour_start).value // 1_000_000)
    return pd.DataFrame({
        "time": pd.to_datetime(t0 + arr["ms"].astype("int64"), unit="ms", utc=True),
        "bid": arr["bid"] / div, "ask": arr["ask"] / div,
        "bid_vol": arr["bid_vol"].astype("float32"), "ask_vol": arr["ask_vol"].astype("float32"),
    })


def _fetch(url):
    try:
        return fx_fetch.fetch_bytes(url)
    except RuntimeError as e:
        print(f"skip {e}", flush=True)
        return None


def duka_ticks(sym, day, h0, h1):
    """Ticks of the hours [h0, h1) of `day` (UTC). Returns (ticks, hours that could not be fetched)."""
    starts = [pd.Timestamp(day, tz="UTC") + pd.Timedelta(hours=h) for h in range(h0, h1)]
    urls = [DUKA_TICKS.format(sym=sym, y=s.year, m=s.month - 1, d=s.day, h=s.hour) for s in starts]
    with ThreadPoolExecutor(int(os.environ.get("FX_WORKERS", 4))) as ex:
        raws = list(ex.map(_fetch, urls))
    failed = [s.hour for r, s in zip(raws, starts) if r is None]
    parts = [decode_ticks(r, s, divisor(sym)) for r, s in zip(raws, starts) if r is not None]
    parts = [p for p in parts if p is not None and len(p)]
    if not parts:
        return pd.DataFrame(columns=["time", "bid", "ask", "bid_vol", "ask_vol"]), failed
    return pd.concat(parts, ignore_index=True), failed


def duka_candles(sym, kind, period):
    """m1: one month of minute candles; h1: one year of hour candles (bid and ask)."""
    div = divisor(sym)
    today = dt.datetime.now(dt.timezone.utc).date()
    if kind == "m1":
        y, m = map(int, period.split("-"))
        first = dt.date(y, m, 1)
        nxt = dt.date(y + (m == 12), m % 12 + 1, 1)
        starts = [d for d in (first + dt.timedelta(days=i) for i in range((nxt - first).days))
                  if d.weekday() != 5 and d < today]
        url = fx_fetch.URL_M1
    else:
        y = int(period)
        starts = [dt.date(y, m, 1) for m in range(1, 13) if dt.date(y, m, 1) < dt.date(today.year, today.month, 1)]
        url = fx_fetch.URL_H1
    sides = {}
    for side in ("bid", "ask"):
        urls = [url.format(sym=sym, y=s.year, m=s.month - 1, d=s.day, side=side.upper()) for s in starts]
        with ThreadPoolExecutor(int(os.environ.get("FX_WORKERS", 4))) as ex:
            raws = list(ex.map(_fetch, urls))
        if any(r is None for r in raws):
            raise RuntimeError(f"{sym} {kind} {period} {side}: {sum(r is None for r in raws)} files not fetched")
        parts = [fx_fetch.decode(r, s, div) for r, s in zip(raws, starts)]
        parts = [p for p in parts if p is not None and len(p)]
        sides[side] = (pd.concat(parts, ignore_index=True) if parts
                       else pd.DataFrame(columns=["time"] + fx_fetch.OHLC + ["volume"]))
    return fx_fetch.merge_sides(sides["bid"], sides["ask"])


def item_name(item):
    kind, sym, rest = item.split(":", 2)
    if kind == "t":
        day, hours = rest.split(":")
        return f"duka_t_{sym}_{day}_{hours}.parquet"
    return f"duka_{kind}_{sym}_{rest}.parquet"


def run_duka(items, out, tag, budget_min):
    t0, failed = time.time(), []
    for item in items:
        if (time.time() - t0) / 60 > budget_min:
            print(f"budget reached, leaving {item} and the rest for the next run", flush=True)
            break
        kind, sym, rest = item.split(":", 2)
        t = time.time()
        missing = []
        try:
            if kind == "t":
                day, hours = rest.split(":")
                h0, h1 = map(int, hours.split("-"))
                df, missing = duka_ticks(sym, day, h0, h1)
            else:
                df = duka_candles(sym, kind, rest)
        except Exception as e:  # keep going; the next run retries
            print(f"FAIL {item}: {e}", flush=True)
            failed.append(item)
            continue
        path = Path(out) / item_name(item)
        write(df, path)
        miss = path.with_suffix(".missing.txt")
        if missing:  # partial: keep it, list the hours; the planner fetches the item again
            miss.write_text("\n".join(map(str, missing)) + "\n")
            upload(tag, [miss])
            failed.append(item)
        elif tag:  # complete now: drop an old list of missing hours
            subprocess.run(["gh", "release", "delete-asset", tag, miss.name, "-y"], check=False,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        lo = df.iloc[:, 1].min() if len(df) else float("nan")
        hi = df.iloc[:, 1].max() if len(df) else float("nan")
        print(f"OK {item}: {len(df)} rows, first col range {lo:.6g}..{hi:.6g}, "
              f"{path.stat().st_size / 1e6:.2f} MB in {time.time() - t:.0f}s", flush=True)
        upload(tag, [path])
        if tag:
            path.unlink()
    print(f"requests {fx_fetch.STATS['requests']}, retries {fx_fetch.STATS['retries']}, "
          f"{len(failed)} failed: {','.join(failed)}", flush=True)


# ---------------------------------------------------------------- second source

def iso_weeks(month):
    """(iso_year, week) pairs of all ISO weeks that overlap the month."""
    y, m = map(int, month.split("-"))
    d, end = dt.date(y, m, 1), dt.date(y + (m == 12), m % 12 + 1, 1)
    weeks = []
    while d < end:
        iy, iw, _ = d.isocalendar()
        if (iy, iw) not in weeks:
            weeks.append((iy, iw))
        d += dt.timedelta(days=1)
    return weeks


def _text(raw):
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return raw.decode("utf-16")
    return raw.decode("utf-8", errors="replace")


def fxcm_ticks(sym, month):
    """FXCM tick files (one gzip CSV per symbol and week: DateTime, Bid, Ask)."""
    parts = []
    for iy, iw in iso_weeks(month):
        status, raw = http_get(f"https://tickdata.fxcorporate.com/{sym}/{iy}/{iw}.csv.gz", timeout=300)
        if status != 200:
            print(f"fxcm {sym} {iy}-W{iw}: {status}", flush=True)
            if status == 404:
                continue
            raise RuntimeError(f"fxcm {sym} {iy}-W{iw}: {status}")
        d = pd.read_csv(io.StringIO(_text(gzip.decompress(raw))))
        d.columns = [c.strip() for c in d.columns]
        d["time"] = pd.to_datetime(d["DateTime"], format="%m/%d/%Y %H:%M:%S.%f", utc=True)
        parts.append(d[["time", "Bid", "Ask"]].rename(columns={"Bid": "bid", "Ask": "ask"}))
    return pd.concat(parts, ignore_index=True) if parts else None


def exness_ticks(sym, month):
    """Exness tick archive (one zip per symbol and month: Exness, Symbol, Timestamp, Bid, Ask)."""
    y, m = month.split("-")
    status, raw = http_get(f"https://ticks.ex2archive.com/ticks/{sym}/{y}/{m}/Exness_{sym}_{y}_{m}.zip", timeout=600)
    if status != 200:
        print(f"exness {sym} {month}: {status}", flush=True)
        if status == 404:
            return None
        raise RuntimeError(f"exness {sym} {month}: {status}")
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        name = [n for n in z.namelist() if n.lower().endswith(".csv")][0]
        d = pd.read_csv(z.open(name))
    d.columns = [c.strip().strip('"') for c in d.columns]
    d["time"] = pd.to_datetime(d["Timestamp"], utc=True, format="ISO8601")
    return d[["time", "Bid", "Ask"]].rename(columns={"Bid": "bid", "Ask": "ask"})


def to_minutes(ticks, month):
    """Bid/ask minute bars (time = start of the minute, UTC) with tick count and mean spread."""
    y, m = map(int, month.split("-"))
    lo = pd.Timestamp(y, m, 1, tz="UTC")
    hi = pd.Timestamp(y + (m == 12), m % 12 + 1, 1, tz="UTC")
    t = ticks[(ticks.time >= lo) & (ticks.time < hi)].sort_values("time", kind="stable")
    g = t.groupby(t.time.dt.floor("1min"))
    out = pd.DataFrame({
        "bid_open": g.bid.first(), "bid_high": g.bid.max(), "bid_low": g.bid.min(), "bid_close": g.bid.last(),
        "ask_open": g.ask.first(), "ask_high": g.ask.max(), "ask_low": g.ask.min(), "ask_close": g.ask.last(),
        "ticks": g.bid.size().astype("int32"), "spread_mean": (t.ask - t.bid).groupby(t.time.dt.floor("1min")).mean(),
    })
    out.index.name = "time"
    return out.reset_index()


def run_second(source, month, symbols, out, tag):
    fetch = {"fxcm": fxcm_ticks, "exness": exness_ticks}[source]
    for sym in symbols:
        t = time.time()
        try:
            ticks = fetch(sym, month)
        except Exception as e:
            print(f"FAIL second {source} {sym} {month}: {e}", flush=True)
            continue
        if ticks is None or ticks.empty:
            print(f"NONE second {source} {sym} {month}", flush=True)
            continue
        bars = to_minutes(ticks, month)
        path = Path(out) / f"second_{source}_{sym}_{month}.parquet"
        write(bars, path)
        print(f"OK second {source} {sym} {month}: {len(ticks)} ticks -> {len(bars)} minutes, "
              f"median spread {float((ticks.ask - ticks.bid).median()):.6g} in {time.time() - t:.0f}s", flush=True)
        upload(tag, [path])
        if tag:
            path.unlink()


# ---------------------------------------------------------------- misc (raw downloads)

YAHOO = ["6E=F", "6J=F", "6B=F", "6A=F", "6C=F", "6S=F", "6N=F", "DX=F"]
FRED = ["DTWEXAFEGS", "DTWEXBGS", "DTWEXEMEGS"]
WAYBACK = "https://web.archive.org/web/{ts}id_/{url}"
BLS_SCHEDULE = "https://www.bls.gov/schedule/news_release/empsit.htm"
BLS_ARCHIVE = "https://www.bls.gov/bls/news-release/empsit.htm"
RAW_URLS = {
    "fed_fomccalendars.htm": "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm",
    **{f"fed_fomchistorical{y}.htm": f"https://www.federalreserve.gov/monetarypolicy/fomchistorical{y}.htm"
       for y in range(2013, 2021)},
    "fed_h10_summary.htm": "https://www.federalreserve.gov/releases/h10/summary/default.htm",
    "fed_h10_package.zip": "https://www.federalreserve.gov/datadownload/Output.aspx?rel=H10&filetype=zip",
    "bis_tov_api.csv": "https://stats.bis.org/api/v1/data/WS_DER_OTC_TOV/all?format=csv",
    "bis_tov_bulk.zip": "https://data.bis.org/static/bulk/WS_DER_OTC_TOV_csv_flat.zip",
    **{f"bls_schedule_{ts}.htm": WAYBACK.format(ts=ts, url=BLS_SCHEDULE)
       for ts in ("20240615", "20241215", "20250615", "20251215", "20260615", "20261020")},
    "bls_archive_latest.htm": WAYBACK.format(ts="20261020", url=BLS_ARCHIVE),
    "bls_archive_2025.htm": WAYBACK.format(ts="20250701", url=BLS_ARCHIVE),
}


def _save(out, name, raw, tag):
    p = Path(out) / name
    if name.endswith(".csv") and len(raw) > 20e6:
        p = p.with_suffix(".csv.gz")
        raw = gzip.compress(raw)
    p.write_bytes(raw)
    upload(tag, [p])
    return p


def ft_bbdxy(out, tag):
    """Bloomberg Dollar Spot Index (BBDXY) daily from markets.ft.com: the tearsheet page
    holds an internal id, the historical-prices endpoint returns HTML table rows per range."""
    import re
    status, page = http_get("https://markets.ft.com/data/indices/tearsheet/historical?s=BBDXY:IOM", timeout=60)
    print(f"ft page: {status} {len(page)} bytes", flush=True)
    if status != 200:
        return
    _save(out, "ft_bbdxy_page.htm", page, tag)
    m = re.search(rb'xid(?:&quot;|")\s*:\s*(?:&quot;|")(\d+)', page) or re.search(rb'data-mod-symbol-xid="(\d+)"', page)
    if not m:
        print("ft: no xid found", flush=True)
        return
    xid = m.group(1).decode()
    for y in range(2020, 2027):
        url = (f"https://markets.ft.com/data/equities/ajax/get-historical-prices?startDate={y}/01/01"
               f"&endDate={y}/12/31&symbol={xid}")
        status, raw = http_get(url, timeout=60)
        print(f"ft bbdxy {y}: {status} {len(raw)} bytes", flush=True)
        if status == 200 and raw:
            _save(out, f"ft_bbdxy_{y}.json", raw, tag)


def yahoo(out, tag, tickers=None):
    import yfinance as yf
    now = pd.Timestamp.now(tz="UTC").floor("1D")
    for tk in tickers or YAHOO:
        safe = tk.replace("=", "")
        for interval, chunks in (("60m", [dict(period="730d")]),
                                 ("1m", [dict(start=(now - pd.Timedelta(days=d + 7)).date(),
                                              end=(now - pd.Timedelta(days=d)).date()) for d in (0, 7, 14, 21)])):
            parts = []
            for kw in chunks:
                h = None
                for k in range(5):
                    try:
                        h = yf.Ticker(tk).history(interval=interval, auto_adjust=False, prepost=True, **kw)
                        if h is not None and len(h):
                            break
                    except Exception as e:
                        print(f"yahoo {tk} {interval} {kw}: {e}", flush=True)
                    time.sleep(20 * (k + 1))
                if h is not None and len(h):
                    h = h.reset_index()
                    h = h.rename(columns={h.columns[0]: "time"})
                    h["time"] = pd.to_datetime(h["time"], utc=True)
                    parts.append(h)
                time.sleep(3)
            if parts:
                df = pd.concat(parts, ignore_index=True).drop_duplicates("time").sort_values("time")
                df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]
                path = Path(out) / f"yahoo_{interval}_{safe}.parquet"
                write(df, path)
                upload(tag, [path])
                print(f"OK yahoo {tk} {interval}: {len(df)} rows {df.time.min()} .. {df.time.max()}", flush=True)
            else:
                print(f"FAIL yahoo {tk} {interval}: no rows", flush=True)


def misc(out, tag, have=()):
    for name, url in RAW_URLS.items():
        if name in have:
            continue
        status, raw = http_get(url, timeout=300, headers={"Accept": "*/*"})
        print(f"{name}: {status} {len(raw)} bytes", flush=True)
        if status == 200 and raw:
            _save(out, name, raw, tag)
    try:
        ft_bbdxy(out, tag)
    except Exception as e:
        print(f"FAIL ft: {e}", flush=True)
    for sid in FRED:  # last: fred.stlouisfed.org often times out for GitHub runners
        if f"fred_{sid}.csv" in have:
            continue
        for url in (f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}",
                    f"https://fred.stlouisfed.org/series/{sid}/downloaddata/{sid}.csv"):
            status, raw = http_get(url, timeout=90, tries=1)
            print(f"fred {sid}: {status} {len(raw)} bytes  {url}", flush=True)
            if status == 200 and raw:
                _save(out, f"fred_{sid}.csv", raw, tag)
                break


# ---------------------------------------------------------------- probe

def probe_urls():
    d = dt.date(2024, 8, 5)
    return {
        "duka_ticks_EURUSD": DUKA_TICKS.format(sym="EURUSD", y=2024, m=7, d=5, h=10),
        "duka_ticks_CHFJPY": DUKA_TICKS.format(sym="CHFJPY", y=2024, m=7, d=5, h=10),
        "duka_ticks_EURUSD_2015": DUKA_TICKS.format(sym="EURUSD", y=2015, m=0, d=15, h=8),
        "duka_m1_DOLLARIDXUSD": fx_fetch.URL_M1.format(sym="DOLLARIDXUSD", y=2024, m=7, d=5, side="BID"),
        "duka_h1_DOLLARIDXUSD": fx_fetch.URL_H1.format(sym="DOLLARIDXUSD", y=2024, m=7, side="BID"),
        "duka_m1_USA500IDXUSD": fx_fetch.URL_M1.format(sym="USA500IDXUSD", y=2024, m=7, d=5, side="BID"),
        "fxcm_ticks_2024": "https://tickdata.fxcorporate.com/EURUSD/2024/32.csv.gz",
        "fxcm_ticks_2026": "https://tickdata.fxcorporate.com/EURUSD/2026/36.csv.gz",
        "fxcm_ticks_AUDUSD": "https://tickdata.fxcorporate.com/AUDUSD/2024/32.csv.gz",
        "exness_2024_08": "https://ticks.ex2archive.com/ticks/EURUSD/2024/08/Exness_EURUSD_2024_08.zip",
        "exness_2026_09": "https://ticks.ex2archive.com/ticks/EURUSD/2026/09/Exness_EURUSD_2026_09.zip",
        "truefx": "https://www.truefx.com/dev/data/2024/AUGUST-2024/EURUSD-2024-08.zip",
        "histdata": "https://www.histdata.com/download-free-forex-historical-data/?/ascii/tick-data-quotes/eurusd/2024/8",
        "yahoo_chart": "https://query1.finance.yahoo.com/v8/finance/chart/6E=F?interval=60m&range=5d",
        "fred": "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DTWEXAFEGS",
        "bis_api": "https://stats.bis.org/api/v1/data/WS_CBPOL/D.US?startPeriod=2026-01-01&format=csv",
        "fed": "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm",
        "bls": "https://www.bls.gov/schedule/news_release/empsit.htm",
        "bbdxy_yahoo": "https://query1.finance.yahoo.com/v8/finance/chart/%5EBBDXY?interval=1d&range=5d",
        "bbdxy_ft": "https://markets.ft.com/data/indices/tearsheet/historical?s=BBDXY:IOM",
        "bbdxy_investing": "https://www.investing.com/indices/bloomberg-dollar-index-historical-data",
        "bbdxy_wsj": "https://www.wsj.com/market-data/quotes/index/XX/BBDXY/historical-prices",
        "cme": "https://www.cmegroup.com/CmeWS/mvc/Volume/Details/F/58/20241025/P?tradeDate=20241025",
        "_date": str(d),
    }


def probe(out):
    rows = {}
    for name, url in probe_urls().items():
        if name.startswith("_"):
            continue
        t = time.time()
        status, raw = http_get(url, timeout=60, tries=1)
        rows[name] = {"url": url, "status": status, "bytes": len(raw), "sec": round(time.time() - t, 1),
                      "head": raw[:80].hex() if raw else ""}
        print(f"{name:24s} {status!s:6s} {len(raw):>10d} B {rows[name]['sec']:>6.1f}s  {url}", flush=True)
    path = Path(out) / "probe.json"
    path.write_text(json.dumps(rows, indent=1))
    return [path]


# ---------------------------------------------------------------- plan

def plan(request, assets, jobs):
    """Matrix for the duka jobs: items without a release asset, in request order, round robin."""
    req = json.loads(Path(request).read_text())
    have = set()
    if assets and Path(assets).exists():
        have = {line.split("\t")[0] for line in Path(assets).read_text().splitlines() if line.strip()}
    def done(it):
        name = item_name(it)
        return name in have and name.replace(".parquet", ".missing.txt") not in have
    todo = [it for it in req.get("duka", []) if not done(it)] if req.get("duka_on", True) else []
    buckets = [todo[i::jobs] for i in range(jobs)]
    matrix = [{"id": i, "items": ",".join(b)} for i, b in enumerate(buckets) if b]
    second = []
    sec = req.get("second") or {}
    for month in sec.get("months", []):
        syms = [s for s in sec.get("symbols", PAIRS)
                if f"second_{sec['source']}_{s}_{month}.parquet" not in have]
        if syms:
            second.append({"source": sec["source"], "month": month, "symbols": ",".join(syms)})
    print(f"Plan: {len(todo)} duka items open on {len(matrix)} jobs; second: {len(second)} months", file=sys.stderr)
    print(f"matrix={json.dumps(matrix)}")
    print(f"count={len(matrix)}")
    print(f"second={json.dumps(second)}")
    print(f"nsecond={len(second)}")
    print(f"probe={'true' if req.get('probe') else 'false'}")
    print(f"misc={'true' if req.get('misc') else 'false'}")
    print(f"yahoo={'true' if req.get('yahoo') else 'false'}")
    print(f"yahoo_symbols={','.join(req.get('yahoo_symbols', YAHOO))}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=["plan", "probe", "misc", "yahoo", "duka", "second"])
    ap.add_argument("--request", default="fx-index/request.json")
    ap.add_argument("--assets")
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--items", default="")
    ap.add_argument("--budget-min", type=float, default=330)
    ap.add_argument("--source", default="fxcm")
    ap.add_argument("--month", default="")
    ap.add_argument("--symbols", default=",".join(PAIRS))
    ap.add_argument("--out", default="out")
    ap.add_argument("--upload", metavar="TAG", default="")
    args = ap.parse_args()
    if args.task == "plan":
        return plan(args.request, args.assets, args.jobs)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    if args.task == "probe":
        upload(args.upload, probe(args.out))
    elif args.task == "misc":
        have = set()
        if args.assets and Path(args.assets).exists():
            have = {line.split("\t")[0] for line in Path(args.assets).read_text().splitlines() if line.strip()}
        misc(args.out, args.upload, have)
    elif args.task == "yahoo":
        yahoo(args.out, args.upload, [s for s in args.symbols.split(",") if s] if args.symbols != ",".join(PAIRS) else None)
    elif args.task == "duka":
        run_duka([i for i in args.items.split(",") if i], args.out, args.upload, args.budget_min)
    elif args.task == "second":
        run_second(args.source, args.month, [s for s in args.symbols.split(",") if s], args.out, args.upload)


if __name__ == "__main__":
    main()
