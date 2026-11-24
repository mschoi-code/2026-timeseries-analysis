"""E19: marriages seen as rates instead of counts: comparison table, E7 catch-up with age-specific rates, E15 steps on a monthly rate.

Report only; rules are in notes/decisions.md. Run from births/: python -I scripts/rate_view.py   (BIRTHS_WORKERS, default 6)
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
import policy_intervention as PI  # noqa: E402
from kosis_check import MARR_AGES, annual_item, women  # noqa: E402
from population_variable import POP_2534  # noqa: E402
from province_panel import load  # noqa: E402
from province_standardized import standardized, wife_rates  # noqa: E402

warnings.filterwarnings("ignore")
WORKERS = int(os.environ.get("BIRTHS_WORKERS", 6))
SHOW = ["20 - 24세", "25 - 29세", "30 - 34세", "35 - 39세"]
WINDOWS = [(2019, 2022), (2022, 2025), (2023, 2025)]
CFS = {"trend 2015~2019": 2015, "trend 2010~2019": 2010, "hold 2019": None}
RATE = PI.marr / POP_2534.reindex(PI.marr.index)


# --- (a) count vs rate table ---------------------------------------------------------------
def table(rates):
    marr = annual_item("혼인건수__조혼인율*", "혼인건수(건)")
    T = pd.DataFrame({"count": marr, "crude_rate": annual_item("혼인건수__조혼인율*", "조혼인율(천명당)"),
                      "general_rate_wife": load()[2]["전국"],
                      "asmr_wife_2005": standardized({"n": rates}, 2005)["n"], "asmr_wife_2020": standardized({"n": rates}, 2020)["n"]})
    T = T.join(rates[SHOW].add_prefix("wife ")).loc[2010:2025]
    assert T.notna().all().all(), T[T.isna().any(axis=1)]
    ch = pd.DataFrame({f"{a}->{b}": (T.loc[b] / T.loc[a] - 1) * 100 for a, b in WINDOWS})
    return T, ch


# --- (b) E7 with age-specific rates -----------------------------------------------------------
def cf_rates(rates, first, last, years):
    """Counterfactual rates by band: log-linear trend fitted on first..last, or last year's rates held."""
    if first is None:
        return pd.DataFrame([rates.loc[last].values] * len(years), index=years, columns=rates.columns)
    fit_yrs = list(range(first, last + 1))
    out = {}
    for b in rates.columns:
        slope, icpt = np.polyfit(fit_yrs, np.log(rates.loc[fit_yrs, b].values), 1)
        out[b] = np.exp(icpt + slope * np.array(years))
    return pd.DataFrame(out, index=years)


def catchup(rates):
    W = women()[MARR_AGES]
    implied = (rates * W.loc[rates.index]).sum(axis=1) / 1000  # marriages implied by wife rates 15~54
    rows, annual = [], []
    for name, first in CFS.items():
        yrs = list(range(2020, 2026))
        cf = cf_rates(rates, first, 2019, yrs)
        gap = ((rates.loc[yrs] - cf) * W.loc[yrs]).sum(axis=1) / 1000
        vyrs = list(range(2017, 2020))
        vcf = cf_rates(rates, None if first is None else first - 3, 2016, vyrs)
        valid = (implied.loc[vyrs].sum() / ((vcf * W.loc[vyrs]).sum(axis=1) / 1000).sum() - 1) * 100
        cum = gap.cumsum()
        rows.append(dict(cf=name, deficit_2020_22=gap.loc[2020:2022].sum(), surplus_2023_25=gap.loc[2023:2025].sum(),
                         cum_2025=cum.loc[2025], valid_2017_19_pct=valid))
        annual += [dict(cf=name, year=y, gap=gap[y], cf_asmr_2005=float((cf.loc[y] * W.loc[2005] / W.loc[2005].sum()).sum()))
                   for y in yrs]
    # E7's count counterfactuals on the same annual span (2020~2025), for the side-by-side
    e7 = pd.read_csv(REP / "19_catchup_annual.csv")
    e7 = e7[e7.year <= 2025]
    for name, g in e7.groupby("cf"):
        rows.append(dict(cf=f"E7 {name}", deficit_2020_22=g[g.year <= 2022].gap.sum(), surplus_2023_25=g[g.year >= 2023].gap.sum(),
                         cum_2025=g.gap.sum(), valid_2017_19_pct=np.nan))
    marr = annual_item("혼인건수__조혼인율*", "혼인건수(건)")
    cover = (implied / marr.reindex(implied.index)).loc[2010:2025]
    return pd.DataFrame(rows), pd.DataFrame(annual), cover


# --- (c) E15 step dummies on the monthly rate -------------------------------------------------
def one(job):
    y_kind, tau, covid_on = job
    y = np.log(PI.marr if y_kind == "count" else RATE)
    X = pd.DataFrame({"weekdays": PI.log_wd, "step": (y.index >= tau).astype(float)}, index=y.index)
    if covid_on:
        X["covid"] = PI.covid
    r = fit(y, X, PI.MARR_SPEC)
    return dict(y=y_kind, tau=tau, covid=covid_on, b=r.params["step"], t=r.params["step"] / r.bse["step"], llf=r.llf,
                converged=r.mle_retvals["converged"])


def steps():
    assert RATE.notna().all(), "women 25~34 must cover every month of the marriage series"
    jobs = [(k, tau, c) for k in ("count", "rate") for c in (True, False) for tau in list(PI.PLACEBO) + list(PI.SCAN)]
    with ProcessPoolExecutor(WORKERS) as ex:
        A = pd.DataFrame(list(ex.map(one, jobs, chunksize=4)))
    # the count rows must reproduce E15 before the rate rows mean anything
    old = pd.read_csv(REP / "27_all_dates.csv", parse_dates=["tau"])
    old = old[old.series == "marriages"].set_index(["tau", "covid"]).b
    new = A[A.y == "count"].set_index(["tau", "covid"]).b
    assert (new - old.reindex(new.index)).abs().max() < 1e-6, "E15 not reproduced"
    rows = []
    for (k, c), g in A.groupby(["y", "covid"], sort=False):
        null = g[g.tau.isin(PI.PLACEBO)].t.abs()
        scan = g[g.tau.isin(PI.SCAN)]
        best = scan.loc[scan.llf.idxmax()]
        for name, d in PI.POLICY.items():
            row = scan[scan.tau == pd.Timestamp(d)].iloc[0]
            rows.append(dict(y=k, covid=c, policy=name, effect_pct=(np.exp(row.b) - 1) * 100, t=row.t,
                             placebo_p=float((null >= abs(row.t)).mean()), data_best_tau=best.tau.date(),
                             data_best_effect_pct=(np.exp(best.b) - 1) * 100, unconverged=int((~g.converged).sum())))
    return A, pd.DataFrame(rows)


def main():
    rates = wife_rates()["전국"]
    T, ch = table(rates)
    S, A, cover = catchup(rates)
    allD, P = steps()
    T.to_csv(REP / "31_count_vs_rate.csv")
    ch.to_csv(REP / "31_count_vs_rate_changes.csv")
    S.to_csv(REP / "31_catchup_summary.csv", index=False)
    A.to_csv(REP / "31_catchup_annual.csv", index=False)
    allD.to_csv(REP / "31_steps_all_dates.csv", index=False)
    P.to_csv(REP / "31_steps_policy.csv", index=False)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 20)
    print("implied marriages (wife rates 15~54 x women) / actual count:", cover.round(3).to_dict())
    print("\n% change, count vs rates\n", ch.round(1).to_string())
    print("\ncatch-up, marriages (rate counterfactuals vs E7 count counterfactuals, 2020~2025)\n", S.round(0).to_string(index=False))
    print(A.pivot(index="year", columns="cf", values="gap").round(0))
    print("\nstep dummies, count vs rate (women 25~34)\n", P.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
