"""E4: 1~5-year-ahead annual birth errors with the publication lag, current model against model C. Report only.

December origins, data known to o-2, marriages after that = same month a year earlier, population as in
population_variable.frames (vintage proxy before 2023-12). Rules are in notes/decisions.md.
Run from sarimax_etc/: python -I scripts/long_backtest.py
"""
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))  # appended last: scripts run with python -I
sys.path.append(str(ROOT / "scripts"))
from common import REP  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from population_variable import WINDOWS, path  # noqa: E402
from realtime import D0, D1, PUB_LAG, level, mo  # noqa: E402

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)
ORIGINS = [pd.Timestamp(f"{yr}-12-01") for yr in range(2015, 2025)]
PRIMARY = range(2015, 2021)  # every horizon 1~5 observed; the origins of the 05 backtest
ACTUAL = level.groupby(level.index.year).sum()
LAST_FULL = 2025


def models():
    out = []
    for d, spec in (("d=1", D1), ("d=0", D0)):
        out.append(dict(name=f"{d} | weekdays only", spec=spec, kind="none"))  # frames() then keeps weekdays alone
        for lo, hi in WINDOWS:
            for kind in ("current", "C"):
                out.append(dict(name=f"{d} | {lo}~{hi} | {kind}", spec=spec, kind=kind, lo=lo, hi=hi))
    return out


def job(args):
    model, o = args
    r, f = path(model, o, o, months=60)
    assert r.mle_retvals["converged"], (model["name"], o)
    ann = f.groupby(f.index.year).sum()
    assert list(ann.index) == list(range(o.year + 1, o.year + 6)) and (f.groupby(f.index.year).size() == 12).all()
    return [dict(name=model["name"], origin=o.year, ahead=yr - o.year, forecast=v, actual=ACTUAL[yr], error=(v / ACTUAL[yr] - 1) * 100) for yr, v in ann.items() if yr <= LAST_FULL]


def main():
    jobs = [(mdl, o) for mdl in models() for o in ORIGINS]
    with ProcessPoolExecutor(8) as ex:
        rows = [r for part in ex.map(job, jobs) for r in part]
    for o in ORIGINS:  # benchmark: the last 12 published months, repeated
        known_to = o - mo(PUB_LAG)
        last12 = level[known_to - mo(11):known_to].sum()
        rows += [dict(name="last 12 published months repeated", origin=o.year, ahead=yr - o.year, forecast=last12, actual=ACTUAL[yr], error=(last12 / ACTUAL[yr] - 1) * 100)
                 for yr in range(o.year + 1, min(o.year + 5, LAST_FULL) + 1)]
    B = pd.DataFrame(rows)
    B.to_csv(REP / "16_backtest_annual.csv", index=False)

    out = []
    for sample, part in (("2015~2020 origins", B[B.origin.isin(PRIMARY)]), ("all origins 2015~2024", B)):
        g = part.assign(abs_error=part.error.abs()).groupby(["name", "ahead"], sort=False)
        out.append(pd.DataFrame({"mae_pct": g.abs_error.mean(), "bias_pct": g.error.mean(), "n": g.size()}).reset_index().assign(sample=sample))
    S = pd.concat(out, ignore_index=True)
    S.to_csv(REP / "16_summary.csv", index=False)
    for sample, g in S.groupby("sample", sort=False):
        print(f"\nmean absolute annual error %, {sample}\n", g.pivot_table(index="name", columns="ahead", values="mae_pct", sort=False).round(1).to_string())
        print(f"mean signed error %, {sample}\n", g.pivot_table(index="name", columns="ahead", values="bias_pct", sort=False).round(1).to_string())
    print("\nsigned error at 5 years by origin\n", B[B.ahead == 5].pivot_table(index="name", columns="origin", values="error", sort=False).round(1).to_string())
    old = pd.read_csv(REP / "05_backtest_annual.csv")
    print("\nreference: 05 backtest without publication lag, mean absolute %\n", old.assign(a=old.error.abs()).pivot_table(index="name", columns="ahead", values="a", sort=False).round(1).to_string())


if __name__ == "__main__":
    main()
