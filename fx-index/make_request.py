"""Writes fx-index/request.json for RDT test 48 (rules in research/48_plan.md of Heimrich1/RDT).

    python3 fx-index/make_request.py            # from the repo root
    python3 fx-index/make_request.py --forward  # forward test T5: DXY CFD 2026-10..12 and DX-Y.NYB
"""
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

G8 = ["EUR", "GBP", "AUD", "NZD", "USD", "CAD", "CHF", "JPY"]
PAIRS = [a + b for i, a in enumerate(G8) for b in G8[i + 1:]]

# Scheduled FOMC decision days 2021-2023 (federalreserve.gov), only to keep them out of the draw.
FOMC_IS = ["2021-01-27", "2021-03-17", "2021-04-28", "2021-06-16", "2021-07-28", "2021-09-22", "2021-11-03",
           "2021-12-15", "2022-01-26", "2022-03-16", "2022-05-04", "2022-06-15", "2022-07-27", "2022-09-21",
           "2022-11-02", "2022-12-14", "2023-02-01", "2023-03-22", "2023-05-03", "2023-06-14", "2023-07-26",
           "2023-09-20", "2023-11-01", "2023-12-13"]
US_HOLIDAYS_TUE_THU = ["2021-11-25", "2022-11-24", "2023-07-04", "2023-11-23"]


def calibration_days(n=3, seed=48):
    """n random Tue-Thu days of the build period 2021-2023 without FOMC days, US holidays and 15 Dec - 5 Jan."""
    days = pd.bdate_range("2021-01-01", "2023-12-31")
    days = days[days.dayofweek.isin([1, 2, 3])]
    bad = set(FOMC_IS) | set(US_HOLIDAYS_TUE_THU)
    days = [d for d in days if d.strftime("%Y-%m-%d") not in bad
            and not ((d.month == 12 and d.day >= 15) or (d.month == 1 and d.day <= 5))]
    pick = np.random.RandomState(seed).choice(len(days), n, replace=False)
    return sorted(days[i].strftime("%Y-%m-%d") for i in pick)


def main():
    items = [f"h1:DOLLARIDXUSD:{y}" for y in range(2021, 2027)]
    items += [f"m1:DOLLARIDXUSD:{m}" for m in ["2024-01", "2024-07", "2025-01", "2025-07", "2026-01", "2026-07"]]
    items += [f"t:{p}:2015-01-15:6-16" for p in PAIRS]                       # SNB, 15 Jan 2015 09:30 CET
    items += [f"t:{p}:2024-08-04:21-24" for p in PAIRS]                      # Aug 2024: Sunday open ...
    items += [f"t:{p}:2024-08-05:0-9" for p in PAIRS]                        # ... and the Tokyo crash
    for day, hours in [("2024-07-05", "12-14"), ("2025-01-10", "12-14"), ("2025-07-03", "12-14"),   # NFP 08:30 NY
                       ("2024-07-31", "18-20"), ("2025-01-29", "18-20"), ("2025-07-30", "18-20")]:  # FOMC 14:00 NY
        items += [f"t:{p}:{day}:{hours}" for p in PAIRS]
    for day in calibration_days():
        items += [f"t:{p}:{day}:0-24" for p in PAIRS]
    # FOMC 14:00 New York is 18:00 UTC in summer, so 18-20 UTC above has no minutes before the decision;
    # the two July FOMC days also get 17-20 UTC (run 13).
    fomc_summer = [f"t:{p}:{day}:17-20" for day in ("2024-07-31", "2025-07-30") for p in PAIRS]
    items += fomc_summer
    # run 1 (2026-10-24): probe, misc, all Dukascopy items. Run 2: second source (Exness; FXCM
    # no longer serves its tick files), Yahoo as its own job, more calendar/benchmark sources;
    # duka_on false so that it does not duplicate the Dukascopy jobs of run 1.
    # Run 3: misc again (FRED last). Run 4: Yahoo only for DX=F (missing after run 1).
    # Run 5: the Dollar Index CFD months again (run 1 lost whole months to single throttled days),
    # now stored partially with a list of missing days; two jobs, the tick items stay with run 1.
    # Run 6: Yahoo has no DX=F any more ("possibly delisted"); the ICE index DX-Y.NYB instead.
    # Run 7: the calibration days would not finish within the budget of run 1. Three more jobs
    # take them from the end (own job ids 10-12, "duka_reverse"); every job now skips items that
    # another run has already uploaded, so the two runs meet in the middle.
    # Run 8: three SNB items (EURNZD, AUDNZD, USDJPY) failed in run 1 without a file; one job (id 20).
    # Run 9: AUDJPY 2024-08-05 failed in run 1; USDJPY and CHFJPY 2023-11-30 have missing hours (run 7).
    # Run 10: a fresh job of run 9 fetched 33 tick hours in 5 minutes, the long-running jobs manage about
    # one hour per minute. Six fresh jobs (ids 30-35) take the calibration days 2022-07-28 and 2022-11-23
    # from the end; run 7 keeps 2023-11-30, run 1 comes from the front. All new jobs skip finished items.
    # Run 11: USDJPY 2023-11-30 lost hour 12 when run 9 fetched the whole day again; partial tick items
    # now fetch only their missing hours and merge them into the file in the release (one job, id 40).
    # Run 12: sweep - everything still open after runs 1-11 (three event items that failed, three
    # calibration items with missing hours); no filter, three jobs (ids 50-52).
    # Run 13: the July FOMC days 17-20 UTC (see above); three jobs (ids 60-62).
    # Run 14: sweep of what is still open after run 13 (no filter, one job, id 70).
    req = {"run": 14, "probe": False, "misc": False, "yahoo": False, "duka_on": True,
           "duka_jobs": 1, "duka_id0": 70,
           "duka": items,
           "second": {"source": "exness", "months": ["2024-08", "2025-04", "2026-09"], "symbols": PAIRS}}
    if "--forward" in sys.argv:
        # Forward test T5 (after 2026-12-18 17:00 New York): the Dollar Index CFD months of the test
        # window and the ICE index from Yahoo; everything else is already in the release.
        fwd = [f"m1:DOLLARIDXUSD:{m}" for m in ["2026-10", "2026-11", "2026-12"]]
        req = {"run": f"T5 {dt.date.today()}", "probe": False, "misc": False, "yahoo": True,
               "yahoo_symbols": ["DX-Y.NYB"], "duka_on": True, "duka_filter": fwd, "duka_jobs": 1,
               "duka": items + fwd}
    Path(__file__).with_name("request.json").write_text(json.dumps(req, indent=1) + "\n")
    print(f"{len(req['duka'])} duka items, calibration days {calibration_days()}, run {req['run']}")


if __name__ == "__main__":
    main()
