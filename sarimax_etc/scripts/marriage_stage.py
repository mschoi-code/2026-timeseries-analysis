"""E11: two-stage forecast, marriages forecast first and fed to the births model, against same-month-last-year filling; report only.

Rules are in notes/decisions.md. Run from sarimax_etc/: python -I scripts/marriage_stage.py   (BIRTHS_WORKERS, default 6)
"""
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))  # appended last: scripts run with python -I
sys.path.append(str(ROOT / "scripts"))
from common import REP, fit  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import population_variable as PV  # noqa: E402
from realtime import D0, D1, PUB_LAG, extend_marr, level, log_wd, m, mo  # noqa: E402

warnings.filterwarnings("ignore")
WORKERS = int(os.environ.get("BIRTHS_WORKERS", 6))
MARR_SPEC = dict(order=(0, 1, 1), seasonal_order=(0, 1, 1, 12), trend=None)
ORIGINS = {pd.Timestamp(f"{yr}-12-01"): 60 for yr in range(2015, 2021)} | {pd.Timestamp("2023-12-01"): 31}
ACTUAL = level.groupby(level.index.year).sum()
ACTUAL_M = m.marr.groupby(m.index.year).sum()


def model_fill(known_to, end):
    """Published marriages, then the marriage SARIMA's forecast."""
    tr = np.log(m.marr["2005":known_to])
    X = pd.DataFrame({"weekdays": log_wd}).reindex(pd.date_range(tr.index[0], end, freq="MS"))
    r = fit(tr, X, MARR_SPEC)
    fut = X.loc[known_to + mo(1):end]
    return pd.concat([m.marr[:known_to], np.exp(r.get_forecast(len(fut), exog=fut).predicted_mean)])


def actual_fill(known_to, end):
    """Upper bound, not a forecast: the marriages that really happened."""
    return extend_marr(min(end, m.index[-1]), end)


FILLS = {"same month last year": extend_marr, "marriage model": model_fill, "actual (upper bound)": actual_fill}


def job(args):
    model, o, months, fill = args
    PV.extend_marr = FILLS[fill]  # frames() reads it at call time
    r, f = PV.path(model, o, o, months=months)
    assert r.mle_retvals["converged"], (model["name"], o, fill)
    return dict(name=model["name"], origin=o, months=months, fill=fill, forecast=f)


def main():
    models = [dict(name=f"{d} | {lo}~{hi}", spec=spec, kind="current", lo=lo, hi=hi) for d, spec in (("d=1", D1), ("d=0", D0)) for lo, hi in ((12, 35), (9, 32))]
    jobs = [(mdl, o, n, fill) for mdl in models for o, n in ORIGINS.items() for fill in FILLS]
    with ProcessPoolExecutor(WORKERS) as ex:
        res = list(ex.map(job, jobs))

    annual, monthly = [], []
    for r in res:
        f = r["forecast"]
        if r["months"] == 60:
            for yr, v in f.groupby(f.index.year).sum().items():
                annual.append(dict(name=r["name"], fill=r["fill"], origin=r["origin"].year, ahead=yr - r["origin"].year, error=(v / ACTUAL[yr] - 1) * 100))
        else:
            a = level[f.index]
            ape = (np.abs(f - a) / a * 100).values
            monthly.append(dict(name=r["name"], fill=r["fill"], all_31=ape.mean(), h1_12=ape[:12].mean(), h13_24=ape[12:24].mean(), h25_31=ape[24:].mean()))
    A = pd.DataFrame(annual)
    M = pd.DataFrame(monthly)

    old = pd.read_csv(REP / "16_backtest_annual.csv")
    old = old[old.name.isin([f"{x['name']} | current" for x in models])].assign(name=lambda d: d.name.str.replace(" | current", ""))
    chk = A[A.fill == "same month last year"].merge(old, on=["name", "origin", "ahead"], suffixes=("", "_old"))
    assert len(chk) == 4 * 6 * 5 and (chk.error - chk.error_old).abs().max() < 1e-6, "16 not reproduced"

    mrows = []  # the fill itself: annual marriages 1~5 years ahead
    for o in [o for o, n in ORIGINS.items() if n == 60]:
        known_to, end = o - mo(PUB_LAG), o + mo(60)
        for fill in ("same month last year", "marriage model"):
            s = FILLS[fill](known_to, end)[o + mo(1):end]
            for yr, v in s.groupby(s.index.year).sum().items():
                mrows.append(dict(fill=fill, origin=o.year, ahead=yr - o.year, error=(v / ACTUAL_M[yr] - 1) * 100))
    MR = pd.DataFrame(mrows)

    A.to_csv(REP / "23_annual_errors.csv", index=False)
    M.to_csv(REP / "23_rebound_31m.csv", index=False)
    MR.to_csv(REP / "23_marriage_fill_errors.csv", index=False)
    pd.set_option("display.width", 220)
    print("reproduced 16 for same-month filling (120 cells)")
    print("\nannual births, mean absolute error %, December 2015~2020 origins\n",
          A.assign(a=A.error.abs()).pivot_table(index=["name", "fill"], columns="ahead", values="a", sort=False).round(1).to_string())
    print("\nannual births, mean signed error %\n", A.pivot_table(index=["name", "fill"], columns="ahead", values="error", sort=False).round(1).to_string())
    print("\nannual marriages from the fill itself, mean absolute / signed error %\n",
          MR.assign(a=MR.error.abs()).pivot_table(index="fill", columns="ahead", values=["a", "error"], sort=False).round(1).to_string())
    print("\n2023-12 origin, 31 months, MAPE % by horizon\n", M.set_index(["name", "fill"]).round(2).to_string())


if __name__ == "__main__":
    main()
