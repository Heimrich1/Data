"""Earnings-Kalender aus den Nachrichten-Fenstern (Panel 2, T3a).

Übernommen aus src/rdt/earnings.py des Forschungscodes (ohne den Datei-Cache).

Festgelegt vor jeder Renditeansicht:
- Ergebnis-Schlagzeile: höchstens 2 Symbole, Muster "Q1-Q4/FY ... EPS/Loss/FFO ... $Zahl",
  keine Ausblicke/Vorschauen ("Sees", "Guidance", "Preview", "Expect", ...).
- Ereignis je Symbol: früheste Ergebnis-Schlagzeile eines Clusters (Schlagzeilen
  innerhalb von 20 Tagen gehören zu derselben Meldung).
- Reaktionssitzung d: Schlagzeile vor 09:30 an Handelstag D -> d = D (BMO);
  ab 16:00 -> d = nächster Handelstag (AMC).
- Überraschung: "EPS $A Beats/Misses $E Estimate" bzw. "$A vs $E" -> A, E (nur Merkmal).
- PIT (vorab bekannt): Sammelartikel "Earnings Scheduled For ..." oder Vorschau
  ("To Report", "Will Report", "Earnings Preview", "Ahead Of ... Earnings") mit dem Symbol,
  erschienen vor 15:30 New York am Handelstag vor d und höchstens 21 Tage alt.
"""

from __future__ import annotations

import re
from datetime import date, datetime, time, timedelta

import numpy as np
import pandas as pd

from zoneinfo import ZoneInfo

NY = ZoneInfo("America/New_York")

RESULT = re.compile(
    r"\b(Q[1-4]|FQ[1-4]|FY\s?\d{0,4})\b.{0,40}?\b(?:Adj\.?\s+|GAAP\s+|Core\s+|Diluted\s+)?"
    r"(EPS|Loss Per Share|Earnings Per Share|FFO|AFFO)\b.{0,8}?\$\s?\(?-?[\d.]+", re.I)
NOT_RESULT = re.compile(
    r"\b(Sees|Guidance|Guides|Outlook|Preview|Expect\w*|Ahead Of|Scheduled|To Report|Will Report|"
    r"Option|Analyst|Price Target|Upgrade|Downgrade|Recap|Consensus|Raises|Lowers|Reaffirms|Affirms)\b", re.I)
SURPRISE = re.compile(
    r"\bEPS\s+\$\s?(\(?-?[\d.]+\)?)\s+(?:Beats|Misses|Up From|Down From|vs\.?)\s+\$\s?(\(?-?[\d.]+\)?)", re.I)
INLINE = re.compile(r"\bEPS\s+\$\s?(\(?-?[\d.]+\)?),?\s+Inline", re.I)
SCHEDULE = re.compile(
    r"Earnings Scheduled For|To Report|Will Report|Earnings Preview|Ahead Of .{0,40}Earnings|Earnings Outlook", re.I)
OPEN, CLOSE, PIT_CUTOFF = time(9, 30), time(16, 0), time(15, 30)


def _num(s: str) -> float:
    s = s.replace("$", "").strip()
    neg = s.startswith("(") or s.startswith("-")
    v = float(s.strip("()-").rstrip("."))
    return -v if neg else v


def result_headlines(h: pd.DataFrame) -> pd.DataFrame:
    m = h["h"].map(lambda x: bool(RESULT.search(x)) and not NOT_RESULT.search(x)) & h["s"].map(len).between(1, 2)
    r = h[m].explode("s").rename(columns={"s": "symbol"})
    def surprise(txt):
        a = SURPRISE.search(txt)
        if a and re.search(r"Beats|Misses|vs", a.group(0), re.I):
            try:
                return _num(a.group(1)), _num(a.group(2))
            except ValueError:
                return np.nan, np.nan
        b = INLINE.search(txt)
        if b:
            v = _num(b.group(1))
            return v, v
        return np.nan, np.nan
    sv = r["h"].map(surprise)
    r["eps_actual"] = [x[0] for x in sv]
    r["eps_est"] = [x[1] for x in sv]
    return r


def build_events(h: pd.DataFrame, trading_days: list[date]) -> pd.DataFrame:
    """Ein Ereignis je Symbol und Meldung mit Reaktionssitzung d, BMO/AMC, Überraschung, PIT."""
    pos = {d: i for i, d in enumerate(trading_days)}
    res = result_headlines(h).sort_values(["symbol", "t"])
    events = []
    for sym, g in res.groupby("symbol"):
        last = None
        for row in g.itertuples():
            if last is not None and (row.t - last) < pd.Timedelta(days=20):
                continue
            last = row.t
            D = row.t.date()
            if D not in pos:
                continue
            if row.t.time() < OPEN:
                d, when = D, "BMO"
            elif row.t.time() >= CLOSE and pos[D] + 1 < len(trading_days):
                d, when = trading_days[pos[D] + 1], "AMC"
            else:
                continue
            events.append({"symbol": sym, "t": row.t, "d": d, "when": when,
                           "eps_actual": row.eps_actual, "eps_est": row.eps_est, "headline": row.h})
    ev = pd.DataFrame(events)
    # PIT: Termin-Artikel mit dem Symbol vor 15:30 am Handelstag vor d, höchstens 21 Tage alt
    sched = h[h["h"].map(lambda x: bool(SCHEDULE.search(x)))].explode("s").rename(columns={"s": "symbol"})
    sched = sched[["symbol", "t"]].sort_values("t")
    by_sym = {s: list(g["t"]) for s, g in sched.groupby("symbol")}
    pit = []
    for e in ev.itertuples():
        i = pos[e.d]
        if i == 0:
            pit.append(False)
            continue
        prev = trading_days[i - 1]
        cutoff = pd.Timestamp(datetime.combine(prev, PIT_CUTOFF), tz=NY)
        lo = cutoff - pd.Timedelta(days=21)
        pit.append(any(lo <= t < cutoff for t in by_sym.get(e.symbol, ())))
    ev["pit"] = pit
    return ev
