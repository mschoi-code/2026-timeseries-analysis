"""E15: step-dummy intervention analysis of policy dates on monthly marriages and births, with a placebo-date null; report only.

Rules are in notes/decisions.md. Run from sarimax_etc/: python -I scripts/policy_intervention.py   (BIRTHS_WORKERS, default 6)
"""
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))  # appended last: scripts run with python -I
sys.path.append(str(ROOT / "scripts"))
from common import REP, fit, weekdays  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from econ_var import vital  # noqa: E402
from realtime import D1, window  # noqa: E402

warnings.filterwarnings("ignore")
WORKERS = int(os.environ.get("BIRTHS_WORKERS", 6))
MARR_SPEC = dict(order=(0, 1, 1), seasonal_order=(0, 1, 1, 12), trend=None)
POLICY = {"P1 2023-01 parent benefit": "2023-01-01", "P2 2024-02 newborn loans, gift deduction": "2024-02-01",
          "P3 2024-07 June 2024 package": "2024-07-01", "P4 2025-01 marriage tax credit, local grants": "2025-01-01"}
PLACEBO = pd.date_range("2010-01-01", "2019-12-01", freq="MS")
SCAN = pd.date_range("2022-01-01", "2025-08-01", freq="MS")
BIRTH_DELAY = 10  # months from a policy date to births it could bring forward
mo = lambda n: pd.DateOffset(months=n)

marr_all = vital("혼인건수(건)")  # from 1981: the newlywed window needs 35 months before 2005
marr = marr_all["2005-01-01":]
births = vital("출생아수(명)")["2005-01-01":]
assert marr.index[-1] == births.index[-1] == pd.Timestamp("2026-08-01")
covid = pd.Series(((marr.index >= "2020-03-01") & (marr.index <= "2022-12-01")).astype(float), index=marr.index)
log_wd = np.log(weekdays(marr.index))


def one(job):
    """job = (series, tau, covid_on, window or None); returns the step coefficient and t."""
    series, tau, covid_on, win = job
    y = np.log(marr if series == "marriages" else births)
    at = tau if series == "marriages" else tau + mo(BIRTH_DELAY)
    X = pd.DataFrame({"weekdays": log_wd, "step": (y.index >= at).astype(float)}, index=y.index)
    if covid_on:
        X["covid"] = covid
    spec = MARR_SPEC
    if win:
        X["newlywed"] = window(*win, marr_all).reindex(y.index)
        spec = D1
    assert X.notna().all().all(), job
    r = fit(y, X, spec)
    return dict(series=series, tau=tau, step_at=at, covid=covid_on, window="" if not win else f"{win[0]}~{win[1]}",
                b=r.params["step"], t=r.params["step"] / r.bse["step"], llf=r.llf, converged=r.mle_retvals["converged"])


def main():
    assert D1["order"] == (0, 1, 1)
    configs = [("marriages", True, None), ("marriages", False, None), ("births", True, (12, 35)), ("births", True, (9, 32))]
    jobs = [(s, tau, c, w) for s, c, w in configs for tau in list(PLACEBO) + list(SCAN)]
    with ProcessPoolExecutor(WORKERS) as ex:
        A = pd.DataFrame(list(ex.map(one, jobs, chunksize=4)))
    A["config"] = A.series + " | covid " + A.covid.map({True: "on", False: "off"}) + " | " + A.window
    A.to_csv(REP / "27_all_dates.csv", index=False)

    rows = []
    for cfg, g in A.groupby("config", sort=False):
        null = g[g.tau.isin(PLACEBO)].t.abs()
        scan = g[g.tau.isin(SCAN)]
        best = scan.loc[scan.llf.idxmax()]
        for name, d in POLICY.items():
            row = scan[scan.tau == pd.Timestamp(d)].iloc[0]
            rows.append(dict(config=cfg, policy=name, step_at=row.step_at.date(), effect_pct=(np.exp(row.b) - 1) * 100, t=row.t,
                             placebo_p=float((null >= abs(row.t)).mean()), placebo_t95=float(null.quantile(0.95)),
                             data_best_tau=best.tau.date(), data_best_effect_pct=(np.exp(best.b) - 1) * 100, unconverged=int((~g.converged).sum())))
    P = pd.DataFrame(rows)
    P.to_csv(REP / "27_policy_dates.csv", index=False)
    pd.set_option("display.width", 220)
    print("step dummy per policy date; placebo_p = share of 120 placebo dates (2010~2019) with |t| at least as large\n")
    print(P.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
