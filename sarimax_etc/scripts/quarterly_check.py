"""Does a quarterly season help? (a) monthly data with s=3, (b) quarterly data with s=4. Report only; see notes/decisions.md."""
import itertools
import json
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))  # appended last: scripts run with python -I
from common import REBOUND_END, REP, SELECT_END, TRAIN_END, design, fit, load_features, load_monthly, mape, rmse, weekdays, women_30_34  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from statsmodels.stats.diagnostic import acorr_ljungbox  # noqa: E402
from statsmodels.tsa.stattools import acf, adfuller, coint, kpss  # noqa: E402

warnings.filterwarnings("ignore")
pd.set_option("display.width", 190)

# --- monthly data
y_m, X_m = load_features()
y_m = y_m[:REBOUND_END]
m = load_monthly()

# --- quarterly data: complete quarters only
counts = m.births.resample("QS").count()
done = counts[counts == 3].index
b_q = m.births.resample("QS").sum().loc[done]
marr_q = m.marr.resample("QS").sum().loc[done]
assert b_q.index[-1] == pd.Timestamp("2026-04-01"), "last complete quarter is 2026Q2"
y_q = np.log(b_q["2005":]).rename("y")
idx_q = pd.date_range(marr_q.index[0], marr_q.index[-1] + pd.DateOffset(months=12), freq="QS")
full_q = marr_q.reindex(idx_q)
nw_q = sum(full_q.shift(k) for k in range(4, 12)) / 2  # 4~11 quarters earlier, the analogue of 12~35 months
X_q = pd.DataFrame({"newlywed": np.log(nw_q), "women": np.log(women_30_34().resample("QS").mean()).reindex(idx_q)}).dropna()
wd = weekdays(pd.date_range(X_q.index[0], X_q.index[-1] + pd.DateOffset(months=2), freq="MS")).resample("QS").sum()
X_q["weekdays"] = np.log(wd.reindex(X_q.index))
assert X_q.notna().all().all() and X_q.index[0] <= y_q.index[0] and X_q.index[-1] == pd.Timestamp("2027-04-01")
bumped = marr_q.copy()  # no leakage: marriages after an origin must not move newlywed up to 4 quarters past it
bumped[bumped.index > "2020-04-01"] *= 2
leak = sum(bumped.reindex(idx_q).shift(k) for k in range(4, 12)) / 2
assert leak[:"2021-04-01"].dropna().equals(nw_q[:"2021-04-01"].dropna()), "quarterly newlywed leaks"

SEL = {"s3": (y_m[:SELECT_END], X_m), "s4": (y_q[:SELECT_END], X_q)}
LB_LAGS = {"s3": [12, 24], "s4": [8, 12]}
PERIOD = {"s3": 3, "s4": 4}
TRACKS = {"proposal": False, "weekdays": True}


def name(spec):
    (p, d, q), (P, D, Q, s) = spec["order"], spec["seasonal_order"]
    return f"({p},{d},{q})({P},{D},{Q}){s}" + ("+c" if spec.get("trend") == "c" else "")


def job(args):
    key, track, spec = args
    y, X = SEL[key]
    base = dict(key=key, track=track, model=name(spec), d=spec["order"][1], spec=spec)
    try:
        r = fit(y, design(X, TRACKS[track], False), spec)
        return {**base, "k": len(r.params), "aic": r.aic, "converged": bool(r.mle_retvals["converged"])}
    except Exception as e:  # a grid cell may fail to start; report it instead of losing the grid
        return {**base, "k": np.nan, "aic": np.nan, "converged": False, "err": str(e)[:60]}


def lb_p(key, spec, calendar, with_x):
    y, X = SEL[key]
    r = fit(y, design(X, calendar, with_x), spec)
    u = pd.Series(r.resid, index=y.index).iloc[r.loglikelihood_burn:]
    n_arma = sum(spec["order"][i] for i in (0, 2)) + sum(spec["seasonal_order"][i] for i in (0, 2))
    return acorr_ljungbox(u, lags=LB_LAGS[key], model_df=n_arma)["lb_pvalue"].round(3).tolist(), r


def pick(g, key, calendar):
    """Lowest AIC, ties within 2 go to fewer parameters, must pass Ljung-Box; if none passes keep the first, flagged."""
    g = g[g.converged].sort_values("aic")
    near = g[g.aic < g.aic.iloc[0] + 2].sort_values(["k", "aic"])
    ranked = pd.concat([near, g[~g.model.isin(near.model)]])
    for n, (_, row) in enumerate(ranked.iterrows()):
        if min(lb_p(key, row.spec, calendar, False)[0]) >= 0.05:
            return row.spec, n
    return ranked.iloc[0].spec, len(ranked)


def candidates(key, G):
    s = PERIOD[key]
    extra = [("proposal-like", dict(order=(0, 1, 1), seasonal_order=(0, 1, 1, s), trend=None)), ("AR(1)-like", dict(order=(1, 0, 0), seasonal_order=(0, 1, 1, s), trend="c"))]
    out = []
    for track, calendar in TRACKS.items():
        named = {}
        for d, g in G[(G.key == key) & (G.track == track)].groupby("d"):
            spec, rejected = pick(g, key, calendar)
            print(f"  {key} {track} d={d}: pick {name(spec)} after {rejected} Ljung-Box rejections of {int(g.converged.sum())} converged fits")
            named.setdefault(name(spec), (f"BJ d={d}", spec))
        for src, spec in extra:
            named.setdefault(name(spec), (src, spec))
        for model, (src, spec) in named.items():
            p0, _ = lb_p(key, spec, calendar, False)
            p1, r1 = lb_p(key, spec, calendar, True)
            out.append(dict(key=key, track=track, model=model, source=src, spec=spec, sarima_lb=p0, sarimax_lb=p1, sarimax_ok=min(p1) >= 0.05,
                            b_nw_2015=r1.params["newlywed"], se_nw_2015=r1.bse["newlywed"]))
    return out


def score(frame):
    return frame.groupby(["track", "model", "kind"], sort=False).agg(mape=("mape", "mean"), rmse=("rmse", "mean")).reset_index()


def eval_s4(cands):
    """Quarterly model: normal 8 Q4 origins re-estimated, rebound frozen at 2023Q4 with 7 origins."""
    rows, rebound, coefs = [], [], []
    level = np.exp(y_q)
    rebound_origins = pd.date_range("2023-10-01", "2025-04-01", freq="QS")
    assert len(rebound_origins) == 7
    for c in cands:
        for with_x in (False, True):
            ex = design(X_q, c["track"] == "weekdays", with_x)
            kind = "SARIMAX" if with_x else "SARIMA"
            for yr in range(2016, 2024):
                r = fit(y_q[:f"{yr - 1}-12"], ex, c["spec"])
                idx = y_q[str(yr)].index
                f = np.exp(r.get_forecast(4, exog=None if ex is None else ex.loc[idx]).predicted_mean)
                rows.append(dict(track=c["track"], model=c["model"], kind=kind, mape=mape(f, level[idx]), rmse=rmse(f, level[idx])))
            res = fit(y_q[:TRAIN_END], ex, c["spec"])
            assert res.mle_retvals["converged"]
            for o in rebound_origins:
                r = res.apply(y_q[:o], exog=None if ex is None else ex.loc[y_q[:o].index])
                idx = pd.date_range(o + pd.DateOffset(months=3), periods=4, freq="QS")
                f = np.exp(r.get_forecast(4, exog=None if ex is None else ex.loc[idx]).predicted_mean)
                rebound.append(dict(track=c["track"], model=c["model"], kind=kind, mape=mape(f, level[idx]), rmse=rmse(f, level[idx])))
            if with_x:
                coefs.append(dict(track=c["track"], model=c["model"], b_newlywed=res.params["newlywed"], se_newlywed=res.bse["newlywed"], b_women=res.params["women"], se_women=res.bse["women"]))
    return score(pd.DataFrame(rows)), score(pd.DataFrame(rebound)), pd.DataFrame(coefs)


def eval_monthly_as_quarters():
    """Existing monthly candidates, forecasts summed to quarters, same origins and same quarters as the s=4 models."""
    cands = json.loads((REP / "01_candidates.json").read_text())
    rows = []
    for c in cands:
        spec = dict(order=tuple(c["spec"]["order"]), seasonal_order=tuple(c["spec"]["seasonal_order"]), trend=c["spec"]["trend"])
        for with_x in (False, True):
            ex = design(X_m, c["track"] == "weekdays", with_x)
            for yr in range(2016, 2024):
                r = fit(y_m[:f"{yr - 1}-12"], ex, spec)
                idx = y_m[str(yr)].index
                f = np.exp(r.get_forecast(12, exog=None if ex is None else ex.loc[idx]).predicted_mean).resample("QS").sum()
                a = b_q.loc[f.index]
                rows.append(dict(track=c["track"], model=c["model"], kind="SARIMAX" if with_x else "SARIMA", mape=mape(f, a), rmse=rmse(f, a)))
    normal = score(pd.DataFrame(rows))

    P = pd.read_csv(REP / "03_rebound_paths.csv", parse_dates=["origin", "target"])
    keep = pd.date_range("2023-12-01", "2025-06-01", freq="3MS")
    assert len(keep) == 7
    P = P[P.origin.isin(keep)].assign(q=lambda f: f.target.dt.to_period("Q"))
    Q = P.groupby(["name", "origin", "q"]).agg(forecast=("forecast", "sum"), actual=("actual", "sum"), n=("actual", "size")).reset_index()
    assert (Q.n == 3).all() and (Q.groupby(["name", "origin"]).size() == 4).all()
    per = Q.groupby(["name", "origin"]).apply(lambda g: pd.Series(dict(mape=mape(g.forecast, g.actual), rmse=rmse(g.forecast, g.actual))), include_groups=False).reset_index()
    reb = per.groupby("name", sort=False)[["mape", "rmse"]].mean().reset_index()
    reb[["model", "track", "kind"]] = reb.name.str.split("|", expand=True)
    return normal, reb.drop(columns="name")


def eval_s3(cands):
    """Monthly data with s=3, on the same monthly metric as steps 2 and 3."""
    rows, reb = [], []
    level = np.exp(y_m)
    origins = pd.date_range(TRAIN_END, "2025-07-01", freq="MS")
    for c in cands:
        for with_x in (False, True):
            ex = design(X_m, c["track"] == "weekdays", with_x)
            kind = "SARIMAX" if with_x else "SARIMA"
            for yr in range(2016, 2024):
                r = fit(y_m[:f"{yr - 1}-12"], ex, c["spec"])
                idx = y_m[str(yr)].index
                f = np.exp(r.get_forecast(12, exog=None if ex is None else ex.loc[idx]).predicted_mean)
                rows.append(dict(track=c["track"], model=c["model"], kind=kind, mape=mape(f, level[idx]), rmse=rmse(f, level[idx])))
            res = fit(y_m[:TRAIN_END], ex, c["spec"])
            for o in origins:
                r = res.apply(y_m[:o], exog=None if ex is None else ex.loc[y_m[:o].index])
                idx = pd.date_range(o + pd.DateOffset(months=1), periods=12, freq="MS")
                f = np.exp(r.get_forecast(12, exog=None if ex is None else ex.loc[idx]).predicted_mean)
                reb.append(dict(track=c["track"], model=c["model"], kind=kind, mape=mape(f, level[idx]), rmse=rmse(f, level[idx])))
    return score(pd.DataFrame(rows)), score(pd.DataFrame(reb))


if __name__ == "__main__":
    a = acf(y_m[:SELECT_END].diff().dropna(), nlags=36)
    print("Q0 monthly D1 ACF through 2015:", {k: round(float(a[k]), 2) for k in (3, 6, 9, 12, 24, 36)})
    q = y_q[:SELECT_END]
    a = acf(q.diff().dropna(), nlags=12)
    print("Q0 quarterly D1 ACF through 2015:", {k: round(float(a[k]), 2) for k in (1, 2, 4, 8, 12)}, "| n", len(q))
    for label_, s in {"log y": q, "D1": q.diff(), "D4": q.diff(4), "D1 D4": q.diff().diff(4)}.items():
        s = s.dropna()
        print(f"Q0 quarterly {label_:6s} n {len(s):3d} ADF c {adfuller(s, regression='c', autolag='AIC')[1]:.3f} ct {adfuller(s, regression='ct', autolag='AIC')[1]:.3f} | KPSS c {kpss(s, regression='c', nlags='auto')[1]:.3f} ct {kpss(s, regression='ct', nlags='auto')[1]:.3f}")

    jobs = []
    for key in SEL:
        specs = [dict(order=(p, d, q_), seasonal_order=(P, 1, Q, PERIOD[key]), trend="c" if d == 0 else None) for d in (1, 0) for p, q_, P, Q in itertools.product(range(3), range(3), range(2), range(2))]
        jobs += [(key, t, s) for t in TRACKS for s in specs]
    with ProcessPoolExecutor(8) as ex:
        G = pd.DataFrame(list(ex.map(job, jobs)))
    print(f"Q1 grid fits {len(G)} | failed {int(G.aic.isna().sum())} | not converged {int((~G.converged).sum())}")

    print("Q2 s=4 candidates")
    c4 = candidates("s4", G)
    print("Q2 s=3 candidates")
    c3 = candidates("s3", G)
    print(pd.DataFrame(c4 + c3)[["key", "track", "model", "source", "sarima_lb", "sarimax_lb", "sarimax_ok", "b_nw_2015", "se_nw_2015"]].round(3).to_string(index=False))

    # --- (b) quarterly vs monthly, same quarters
    n4, r4, co4 = eval_s4(c4)
    nm, rm = eval_monthly_as_quarters()
    lvl = b_q.astype(float)
    nv_n = np.mean([mape(lvl.shift(4)[str(y)], lvl[str(y)]) for y in range(2016, 2024)])
    nv_r = np.mean([mape(lvl.shift(4).loc[pd.date_range(o + pd.DateOffset(months=3), periods=4, freq="QS")], lvl.loc[pd.date_range(o + pd.DateOffset(months=3), periods=4, freq="QS")]) for o in pd.date_range("2023-10-01", "2025-04-01", freq="QS")])
    print(f"Q3 seasonal naive on quarters: normal {nv_n:.2f}, rebound {nv_r:.2f}")
    T4 = n4.merge(r4, on=["track", "model", "kind"], suffixes=("_normal", "_rebound")).assign(data="quarterly s=4")
    TM = nm.merge(rm, on=["track", "model", "kind"], suffixes=("_normal", "_rebound")).assign(data="monthly summed")
    T = pd.concat([TM, T4], ignore_index=True)
    T["d"] = T.model.str[3].astype(int)
    T.to_csv(REP / "07_quarterly_comparison.csv", index=False)
    print("Q4 same quarters, same origins (MAPE %, RMSE births)\n", T[["data", "track", "model", "kind", "d", "mape_normal", "mape_rebound", "rmse_normal", "rmse_rebound"]].round({"mape_normal": 2, "mape_rebound": 2, "rmse_normal": 0, "rmse_rebound": 0}).to_string(index=False))
    print("Q5 exogenous coefficients at 2023Q4\n", co4.round(3).to_string(index=False))
    wide = T.pivot_table(index=["data", "track", "model"], columns="kind", values="mape_rebound", sort=False)
    wide["gain"] = wide.SARIMA - wide.SARIMAX
    print("Q6 rebound gain from exogenous variables\n", wide.round(2).to_string())

    # --- declared rule applied to s=4, for reporting
    ok = pd.DataFrame(c4)[["track", "model", "sarimax_ok"]]
    pick_q = n4[n4.kind == "SARIMAX"].merge(ok, on=["track", "model"])
    pick_q = pick_q[pick_q.sarimax_ok == True].assign(d=lambda f: f.model.str[3].astype(int))  # noqa: E712
    print("Q7 s=4 SARIMAX passing Ljung-Box, normal period\n", pick_q.sort_values("mape").round(2).to_string(index=False))

    # --- how much of the rise do the variables account for under s=4?
    dy = (y_q["2025-07":"2026-06"].mean() - y_q["2023"].mean()) * 100
    dn = (X_q.newlywed["2025-07":"2026-06"].mean() - X_q.newlywed["2023"].mean()) * 100
    dw = (X_q.women["2025-07":"2026-06"].mean() - X_q.women["2023"].mean()) * 100
    print(f"Q8 mean log change 2023 -> 2025Q3~2026Q2: births {dy:.1f}, newlywed {dn:.1f}, women {dw:.1f}")
    for _, c in co4.iterrows():
        a_, b_ = c.b_newlywed * dn, c.b_women * dw
        print(f"   {c.track:9s} {c.model:22s} newlywed {a_ / dy * 100:5.0f}%  women {b_ / dy * 100:5.0f}%  left {(dy - a_ - b_) / dy * 100:5.0f}%")
    ds = y_q[:SELECT_END] - y_q[:SELECT_END].groupby(y_q[:SELECT_END].index.quarter).transform("mean")
    print("Q9 Engle-Granger on quarters through 2015: t %.2f, p %.3f" % coint(ds, X_q.loc[ds.index, ["newlywed", "women"]], trend="c")[:2])

    # --- (a) s=3 on the monthly metric
    n3, r3 = eval_s3(c3)
    T3 = n3.merge(r3, on=["track", "model", "kind"], suffixes=("_normal", "_rebound"))
    T3.to_csv(REP / "07_s3_comparison.csv", index=False)
    print("Q10 monthly data with s=3, monthly metric (compare 3.59~4.05 normal for s=12, d=1)\n", T3.round({"mape_normal": 2, "mape_rebound": 2, "rmse_normal": 0, "rmse_rebound": 0}).to_string(index=False))
