"""Step 3, rebound period: coefficients frozen at 2023-12, 20 origins (2023-12~2025-07), 12-month paths.

Decides nothing: the main model was locked in step 2 (reports/02_locked.json).
"""
import json
import sys
import warnings
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))  # appended last: scripts run with python -I
from common import REBOUND_END, REP, TRAIN_END, design, fit, load_features, mape, rmse  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

warnings.filterwarnings("ignore")
pd.set_option("display.width", 170)
y, X = load_features()
y = y[:REBOUND_END]
level = np.exp(y)
ORIGINS = pd.date_range(TRAIN_END, "2025-07-01", freq="MS")
assert len(ORIGINS) == 20
cands = json.loads((REP / "01_candidates.json").read_text())
lock = json.loads((REP / "02_locked.json").read_text())


def path_rows(name, predict):
    rows = []
    for o in ORIGINS:
        idx = pd.date_range(o + pd.DateOffset(months=1), periods=12, freq="MS")
        f = pd.Series(predict(o, idx), index=idx)
        rows += [dict(name=name, origin=o, target=t, h=h + 1, forecast=f[t], actual=level[t]) for h, t in enumerate(idx)]
    return pd.DataFrame(rows)


def naive(o, idx):
    return level.shift(12).loc[idx].values


paths, coefs, oneshot = [path_rows("seasonal naive|-|baseline", naive)], [], []
for c in cands:
    spec = dict(order=tuple(c["spec"]["order"]), seasonal_order=tuple(c["spec"]["seasonal_order"]), trend=c["spec"]["trend"])
    calendar = c["track"] == "weekdays"
    for with_x in (False, True):
        ex = design(X, calendar, with_x)
        res = fit(y[:TRAIN_END], ex, spec)
        assert res.mle_retvals["converged"]
        name = f"{c['model']}|{c['track']}|{'SARIMAX' if with_x else 'SARIMA'}"

        def predict(o, idx, res=res, ex=ex):
            r = res.apply(y[:o], exog=None if ex is None else ex.loc[y[:o].index])
            assert np.allclose(r.params, res.params), "coefficients must stay frozen"
            return np.exp(r.get_forecast(12, exog=None if ex is None else ex.loc[idx]).predicted_mean).values

        paths.append(path_rows(name, predict))
        if with_x:
            coefs.append(dict(name=name, **{f"b_{k}": res.params[k] for k in ("newlywed", "women")}, **{f"se_{k}": res.bse[k] for k in ("newlywed", "women")}, **{f"p_{k}": res.pvalues[k] for k in ("newlywed", "women")}))

        # one 31-month path from 2023-12, the setting of the proposal slide and the EDA's 7.1%
        idx = y["2024-01":].index
        fc = res.get_forecast(len(idx), exog=None if ex is None else ex.loc[idx])
        ci = fc.conf_int(alpha=0.05)
        oneshot.append(pd.DataFrame(dict(name=name, target=idx, forecast=np.exp(fc.predicted_mean).values, lo=np.exp(ci.iloc[:, 0]).values, hi=np.exp(ci.iloc[:, 1]).values, actual=level[idx].values)))

P = pd.concat(paths, ignore_index=True)
O = pd.concat(oneshot, ignore_index=True)
assert (P.groupby("name").size() == 240).all(), "20 origins x 12 months per model"
first = P[P.origin == ORIGINS[0]].merge(O, on=["name", "target"], suffixes=("", "_o"))
assert len(first) == 12 * (P.name.nunique() - 1) and np.allclose(first.forecast, first.forecast_o), "origin 2023-12 must match the one-shot path"
assert P[(P.name.str.startswith("seasonal")) & (P.target == "2024-01-01") & (P.origin == ORIGINS[0])].forecast.iloc[0] == level["2023-01-01"]
P.to_csv(REP / "03_rebound_paths.csv", index=False)
O.to_csv(REP / "03_rebound_oneshot.csv", index=False)

per_origin = P.groupby(["name", "origin"]).apply(lambda g: pd.Series(dict(mape=mape(g.forecast, g.actual), rmse=rmse(g.forecast, g.actual), bias=((g.forecast / g.actual - 1) * 100).mean())), include_groups=False).reset_index()
S = per_origin.groupby("name", sort=False)[["mape", "rmse", "bias"]].mean()
S["h12_mape"] = P[P.h == 12].groupby("name", sort=False).apply(lambda g: mape(g.forecast, g.actual), include_groups=False)
S["oneshot_31m"] = O.groupby("name", sort=False).apply(lambda g: mape(g.forecast, g.actual), include_groups=False)
S["pi95_outside"] = O.groupby("name", sort=False).apply(lambda g: int(((g.actual < g.lo) | (g.actual > g.hi)).sum()), include_groups=False)
S = S.reset_index()
S[["model", "track", "kind"]] = S.name.str.split("|", expand=True)
nm = pd.read_csv(REP / "02_normal_summary.csv").rename(columns={"mape": "normal_mape"})[["track", "model", "kind", "normal_mape"]]
S = S.merge(nm, how="left", on=["track", "model", "kind"])
S.drop(columns="name").to_csv(REP / "03_rebound_summary.csv", index=False)
print("R1 rebound period, frozen coefficients\n", S[["track", "model", "kind", "normal_mape", "mape", "rmse", "bias", "h12_mape", "oneshot_31m", "pi95_outside"]].round({"normal_mape": 2, "mape": 2, "rmse": 0, "bias": 1, "h12_mape": 2, "oneshot_31m": 1}).to_string(index=False))

# reference values at the true optimum; the EDA's 7.1 came from an optimizer that stopped early, 18.0 and 19/31 are unaffected
g = S.set_index(["track", "model", "kind"])
for key, want in {("proposal", "(0,1,1)(0,1,1)12", "SARIMA"): 18.04, ("proposal", "(0,1,1)(0,1,1)12", "SARIMAX"): 16.38, ("proposal", "(1,0,0)(0,1,1)12+c", "SARIMAX"): 6.51}.items():
    assert abs(g.oneshot_31m[key] - want) < 0.02, (key, g.oneshot_31m[key], want)
assert g.pi95_outside[("proposal", "(0,1,1)(0,1,1)12", "SARIMA")] == 19
print("R2 reference values reproduce, including the slide's 19 of 31 months outside the interval")

pd.DataFrame(coefs).to_csv(REP / "03_coefficients.csv", index=False)
print("R3 coefficients frozen at 2023-12\n", pd.DataFrame(coefs).round(3).to_string(index=False))

# RQ2: same order and calendar term, with and without the two exogenous variables
wide = per_origin.assign(key=lambda f: f.name.str.rsplit("|", n=1).str[0], kind=lambda f: f.name.str.rsplit("|", n=1).str[1]).pivot_table(index=["key", "origin"], columns="kind", values="mape").dropna(subset=["SARIMA", "SARIMAX"])
rq2 = wide.groupby("key", sort=False).apply(lambda f: pd.Series(dict(sarima=f.SARIMA.mean(), sarimax=f.SARIMAX.mean(), gain=(f.SARIMA - f.SARIMAX).mean(), wins_of_20=int((f.SARIMAX < f.SARIMA).sum()))), include_groups=False)
rq2.to_csv(REP / "03_rq2.csv")
print("R4 RQ2 by order (overlapping origins, so wins are not independent)\n", rq2.round(2).to_string())

main_name = f"{lock['main']['model']}|{lock['main']['track']}|SARIMAX"
by_year = P.assign(year=P.target.dt.year).groupby(["name", "year"]).apply(lambda f: mape(f.forecast, f.actual), include_groups=False).unstack().round(1)
print("R5 MAPE by target year\n", by_year.to_string())
print("R6 main", main_name, "| by horizon:", {int(h): round(mape(f.forecast, f.actual), 1) for h, f in P[P.name == main_name].groupby("h")})
