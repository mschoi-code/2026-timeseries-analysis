"""Replace the marriage count and women 30~34 with a marriage rate and an age-weighted population; report only.

Rules are in notes/decisions.md. Run from sarimax_etc/: python -I scripts/population_variable.py
"""
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
from statsmodels.stats.diagnostic import acorr_ljungbox  # noqa: E402
from kosis_check import AGES, national_rates, women  # noqa: E402
from realtime import BLOCKS, D0, D1, NORMAL, PUB_LAG, VINTAGE, extend_marr, level, log_w, log_wd, m, mase_scale, metrics, mo, window, y  # noqa: E402

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 40)
WINDOWS = ((12, 35), (9, 32), (14, 37))
OFFSET = ("C", "E")  # age-weighted population enters with its coefficient fixed at 1
RATES = national_rates()[AGES]
ANNUAL = women()


def to_monthly(annual):
    """Annual values on July 1, linear between: the same rule as common.women_30_34."""
    s = pd.Series(annual.values, index=pd.to_datetime([f"{yr}-07-01" for yr in annual.index]))
    return s.reindex(pd.date_range(s.index[0], s.index[-1], freq="MS")).interpolate("linear")


BANDS = {a: to_monthly(ANNUAL[a].dropna()) for a in AGES}
POP_2534 = BANDS["25 - 29세"] + BANDS["30 - 34세"]


def weight_year(o):
    """Latest final age-specific rates published by origin o: final birth statistics come out in August."""
    return o.year - 1 if o.month >= 9 else o.year - 2


def exposure(year):
    return np.log(sum(RATES.loc[year, a] * BANDS[a] for a in AGES) / 1000)


def safe(log_series, o, known_to, idx):
    """Same vintage proxy as realtime.women_series: before the file's vintage, extend the path known at the origin."""
    out = log_series.reindex(idx).copy()
    if o >= VINTAGE:
        return out
    slope = (log_series[known_to] - log_series[known_to - mo(12)]) / 12
    future = idx[idx > known_to]
    out[future] = log_series[known_to] + slope * np.arange(1, len(future) + 1)
    return out


def rate(lo, hi, marr, log_pop):
    """Log of marriages in the window over women 25~34 in the same window."""
    idx = pd.date_range(marr.index[0], marr.index[-1] + mo(lo), freq="MS")
    pop = np.exp(log_pop).reindex(idx)
    return window(lo, hi, marr) - np.log(sum(pop.shift(k) for k in range(lo, hi + 1)) / (hi - lo + 1))


def frames(model, o, weight_origin, months=12):
    known_to = o - mo(PUB_LAG)
    end = o + mo(months)
    idx = pd.date_range(y.index[0], end, freq="MS")
    cols = {"weekdays": log_wd.reindex(idx)}
    kind = model["kind"]
    ext = extend_marr(known_to, end)
    if kind == "current":
        cols["marr"] = window(model["lo"], model["hi"], ext).reindex(idx)
    if kind in ("A", "B", "C"):
        pop_idx = pd.date_range(POP_2534.index[0], end, freq="MS")
        cols["rate"] = rate(model["lo"], model["hi"], ext, safe(np.log(POP_2534), o, known_to, pop_idx)).reindex(idx)
    if kind in ("current", "A"):
        cols["women"] = safe(log_w, o, known_to, idx)
    expo = safe(exposure(weight_year(weight_origin)), o, known_to, idx)
    if kind in ("B", "D"):
        cols["expo"] = expo
    ex = pd.DataFrame(cols)
    assert ex.notna().all().all(), (model["name"], o)
    return ex, expo if kind in OFFSET else 0.0, known_to


def path(model, o, weight_origin, res0=None, months=12):
    ex, offset, known_to = frames(model, o, weight_origin, months)
    target = (y - offset).dropna() if model["kind"] in OFFSET else y
    tr = target[:known_to]
    r = fit(tr, ex, model["spec"]) if res0 is None else res0.apply(tr, exog=ex.loc[tr.index])
    fut = ex.loc[known_to + mo(1):]
    pred = r.get_forecast(len(fut), exog=fut).predicted_mean
    f = np.exp(pred + (offset.reindex(fut.index) if model["kind"] in OFFSET else 0.0))
    return r, f[pd.date_range(o + mo(1), periods=months, freq="MS")]


def evaluate(model):
    rows = []
    for o in NORMAL:
        _, f = path(model, o, o)
        rows.append(dict(period="official normal", o=o, **metrics(f, mase_scale(o - mo(PUB_LAG)))))
    info = {}
    for label, start in BLOCKS.items():
        o0 = pd.Timestamp(start)
        scale = mase_scale(o0 - mo(PUB_LAG))
        res0 = None
        for o in pd.date_range(o0, periods=20, freq="MS"):
            if res0 is None:
                res0, f = path(model, o, o0)
                if label.startswith("rebound"):
                    assert res0.mle_retvals["converged"], model["name"]
                    info = dict(aic=res0.aic, **{f"b_{c}": res0.params[c] for c in ("rate", "marr", "women", "expo") if c in res0.params.index},
                                **{f"se_{c}": res0.bse[c] for c in ("rate", "marr", "women", "expo") if c in res0.bse.index})
            else:
                _, f = path(model, o, o0, res0=res0)
            rows.append(dict(period=label, o=o, **metrics(f, scale)))

    # residuals through 2015 only, the same check the main-model rule uses
    o = pd.Timestamp("2015-12-01")
    ex, offset, _ = frames(model, o, o)
    target = (y - offset).dropna() if model["kind"] in OFFSET else y
    tr = target[:"2015-12"]
    r = fit(tr, ex, model["spec"])
    n_arma = sum(model["spec"]["order"][i] for i in (0, 2)) + sum(model["spec"]["seasonal_order"][i] for i in (0, 2))
    u = pd.Series(r.resid, index=tr.index).iloc[r.loglikelihood_burn:]
    lb = acorr_ljungbox(u, lags=[12, 24], model_df=n_arma)["lb_pvalue"].tolist()
    info.update(lb12=lb[0], lb24=lb[1])
    return pd.DataFrame(rows).assign(name=model["name"]), dict(name=model["name"], **info)


def scenario_job(model):
    """2031 births by marriage rate (hold, -5%/year) x population (projected, frozen at 2026-07), refit through 2026-07."""
    last = m.index[-1]
    fut = pd.date_range(last + mo(1), "2031-12-01", freq="MS")
    idx = pd.date_range(y.index[0], "2031-12-01", freq="MS")
    monthly_rate = (m.marr / POP_2534)[last - mo(11):last]
    expo = exposure(2025)
    kind = model["kind"]
    base, out = None, []
    for rname, g in (("hold", 0.0), ("decline", -0.05)):
        for pname, frozen in (("projected", False), ("frozen", True)):
            pop = POP_2534.copy()
            if frozen:
                pop[pop.index > last] = pop[last]
            ext = m.marr.copy()
            for i, t in enumerate(fut):
                ext[t] = monthly_rate[monthly_rate.index.month == t.month].iloc[0] * (1 + g) ** (i // 12 + 1) * pop[t]
            hold_at = lambda s: s.where(s.index <= last, s[last]) if frozen else s
            cols = {"weekdays": log_wd.reindex(idx)}
            if kind == "current":
                cols["marr"] = window(model["lo"], model["hi"], ext).reindex(idx)
            if kind in ("A", "B", "C"):
                cols["rate"] = rate(model["lo"], model["hi"], ext, np.log(pop)).reindex(idx)
            if kind in ("current", "A"):
                cols["women"] = hold_at(log_w).reindex(idx)
            if kind in ("B", "D"):
                cols["expo"] = hold_at(expo).reindex(idx)
            X = pd.DataFrame(cols)
            offset = hold_at(expo).reindex(idx) if kind in OFFSET else pd.Series(0.0, index=idx)
            target = y - offset.reindex(y.index)
            if base is None:
                base = fit(target, X, model["spec"])
                coef = {f"b_{c}": base.params[c] for c in ("rate", "marr", "women", "expo") if c in base.params.index}
            r = base.apply(target, exog=X.loc[y.index])
            f = np.exp(r.get_forecast(len(fut), exog=X.loc[fut]).predicted_mean + offset[fut])
            ann = f.groupby(f.index.year).sum()
            ann.loc[2026] += level["2026"].sum()
            out += [dict(name=model["name"], rate=rname, population=pname, year=int(yr), births=float(v), **coef) for yr, v in ann.items()]
    return pd.DataFrame(out)


def main():
    # self-checks: weights, band totals and the publication rule behind every number below
    assert abs(np.exp(exposure(2025)["2025-07-01"]) - (RATES.loc[2025] * ANNUAL.loc[2025, AGES]).sum() / 1000) < 1e-6
    assert weight_year(pd.Timestamp("2023-12-01")) == 2022 and weight_year(pd.Timestamp("2024-08-01")) == 2022 and weight_year(pd.Timestamp("2024-09-01")) == 2023

    models = []
    for d, spec in (("d=1", D1), ("d=0", D0)):
        models.append(dict(name=f"{d} | D expo only", spec=spec, kind="D"))
        models.append(dict(name=f"{d} | E offset only", spec=spec, kind="E"))
        for lo, hi in WINDOWS:
            for kind in ("current", "A", "B", "C"):
                models.append(dict(name=f"{d} | {lo}~{hi} | {kind}", spec=spec, kind=kind, lo=lo, hi=hi))
    with ProcessPoolExecutor(16) as ex:
        res = list(ex.map(evaluate, models))
        sc = list(ex.map(scenario_job, models))
    R = pd.concat([r for r, _ in res], ignore_index=True)
    R.to_csv(REP / "11_by_origin.csv", index=False)
    S = R.groupby(["name", "period"], sort=False)[["mape", "mase", "rmse", "bias", "ape12"]].mean().reset_index()
    S.to_csv(REP / "11_summary.csv", index=False)
    info = pd.DataFrame([i for _, i in res]).set_index("name")
    info.to_csv(REP / "11_info.csv")
    SC = pd.concat(sc, ignore_index=True)
    SC.to_csv(REP / "11_scenarios.csv", index=False)

    # the current model must reproduce the earlier real-time run
    old = pd.read_csv(REP / "09_realtime_summary.csv").set_index(["name", "period"]).mape
    for d in ("d=1", "d=0"):
        for lo, hi in WINDOWS:
            for period in ("official normal", "rebound 2023-12"):
                a = S[(S.name == f"{d} | {lo}~{hi} | current") & (S.period == period)].mape.iloc[0]
                b = old[(f"{d} | {lo}~{hi}", period)]
                assert abs(a - b) < 0.01, (d, lo, hi, period, a, b)
    print("reproduction of 09_realtime_summary: ok")

    wide = S.pivot_table(index="name", columns="period", values="mape", sort=False)
    T = pd.DataFrame({"normal": wide["official normal"], "rebound": wide["rebound 2023-12"]})
    T["normal_mase"] = S[S.period == "official normal"].set_index("name").mase
    T["rebound_mase"] = S[S.period == "rebound 2023-12"].set_index("name").mase
    T = T.join(info)
    y31 = SC[SC.year == 2031].pivot_table(index="name", columns=["rate", "population"], values="births", sort=False)
    T["structure_pct"] = (y31[("hold", "projected")] / y31[("hold", "frozen")] - 1) * 100
    T["structure_people"] = y31[("hold", "projected")] - y31[("hold", "frozen")]
    T["behavior_people"] = y31[("decline", "frozen")] - y31[("hold", "frozen")]
    T["births_2031_hold"] = y31[("hold", "projected")]
    fixed = exposure(2025)
    T["accounting_pct"] = (np.exp(fixed["2031-07-01"] - fixed["2026-07-01"]) - 1) * 100
    T.to_csv(REP / "11_table.csv")
    cols = ["normal", "rebound", "normal_mase", "rebound_mase", "aic", "b_rate", "se_rate", "b_marr", "b_women", "b_expo", "se_expo", "lb12", "lb24", "structure_pct", "structure_people", "behavior_people", "births_2031_hold"]
    print(T[[c for c in cols if c in T.columns]].round(2).to_string())
    print("accounting_pct", round(T.accounting_pct.iloc[0], 2))
    blocks = S[S.period.isin(BLOCKS)].pivot_table(index="name", columns="period", values="mape", sort=False)[list(BLOCKS)]
    print("\nequal blocks of 20 frozen decisions: MAPE\n", blocks.round(2).to_string())


if __name__ == "__main__":
    main()
