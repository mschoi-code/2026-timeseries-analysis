"""Experiments A~F on what drives the rebound and how far the d=0 result can be trusted. Report only; see notes/decisions.md."""
import json
import os
import sys
import warnings
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))  # appended last: scripts run with python -I
from common import DATA, REBOUND_END, REP, TRAIN_END, fit, load_features, load_monthly, mape, newlywed, weekdays  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.api as sm  # noqa: E402
from statsmodels.tsa.stattools import coint  # noqa: E402

warnings.filterwarnings("ignore")
pd.set_option("display.width", 190)
pd.set_option("display.max_columns", 40)

LOCK = json.loads((REP / "02_locked.json").read_text())
SPECS = {(c["track"], c["model"]): c["spec"] for c in json.loads((REP / "01_candidates.json").read_text())}


def locked_spec(d):
    v = LOCK["best_each_d"][d]
    s_ = SPECS[(v["track"], v["model"])]
    return dict(order=tuple(s_["order"]), seasonal_order=tuple(s_["seasonal_order"]), trend=s_["trend"])


D0, D1 = locked_spec("0"), locked_spec("1")
assert all(v["track"] == "weekdays" for v in LOCK["best_each_d"].values()), "experiments assume the weekday regressor"
ORDERS = {"d=0": D0, "d=1": D1}
SUMMARY = pd.read_csv(REP / "03_rebound_summary.csv")


def summary(model, track, kind, col):
    return float(SUMMARY[(SUMMARY.model == model) & (SUMMARY.track == track) & (SUMMARY.kind == kind)].iloc[0][col])


m = load_monthly()
y_all = np.log(m.births[:REBOUND_END]).rename("y")
y, X0 = load_features()
y = y[:REBOUND_END]
level = np.exp(y_all)
ORIGINS = pd.date_range(TRAIN_END, "2025-07-01", freq="MS")

p_ = pd.read_csv(DATA / "성_및_연령별_추계인구_1세별_5세별_전국.csv", header=None, dtype=str)
years = p_.iloc[2, 1:].astype(int).values
P_ = p_.iloc[3:].set_index(0)


def monthly_pop(group):
    s = pd.Series(P_.loc[group].astype(float).values, index=pd.to_datetime([f"{yr}-07-01" for yr in years]))
    return s.reindex(pd.date_range(s.index[0], s.index[-1], freq="MS")).interpolate("linear")


w2534 = monthly_pop("25 - 29세") + monthly_pop("30 - 34세")
lag_pop = sum(w2534.shift(k) for k in range(12, 36)) / 24
nw = newlywed(m.marr)
idx_x = X0.index
X = X0.copy()
X["lag_pop"] = np.log(lag_pop).reindex(idx_x)
X["rate"] = X.newlywed - X.lag_pop
assert X.loc[y.index].notna().all().all()


def window(lo, hi, marr=None):
    """Log of annualized marriages lo~hi months earlier."""
    src = m.marr if marr is None else marr
    idx = pd.date_range(src.index[0], src.index[-1] + pd.DateOffset(months=lo), freq="MS")
    full = src.reindex(idx)
    return np.log(sum(full.shift(k) for k in range(lo, hi + 1)) * 12 / (hi - lo + 1))


assert np.allclose(window(12, 35).dropna().reindex(y.index), np.log(nw).reindex(y.index)), "window must reproduce newlywed"
dy = (y["2025-08":"2026-07"].mean() - y["2023"].mean()) * 100


def change(series):
    return (series["2025-08":"2026-07"].mean() - series["2023"].mean()) * 100


def evaluate(args):
    """Normal (8 December origins, re-estimated) and rebound (frozen at 2023-12, 20 origins)."""
    name, spec, ex, start, covid = args
    y_ = y_all[start:]
    scale = np.exp(y_)

    def regressors(origin):
        if not covid:
            return ex
        out = ex.copy()
        step = ((out.index >= "2020-03-01") & (out.index <= "2022-12-01")).astype(float)
        out["covid"] = np.where(out.index <= pd.Timestamp(origin), step, 0.0)  # no foresight about how long it lasts
        return out if out.covid.any() else ex

    normal = []
    for yr in range(2016, 2024):
        origin = f"{yr - 1}-12-01"
        e = regressors(origin)
        r = fit(y_[:origin], e, spec)
        idx = y_[str(yr)].index
        normal.append(mape(np.exp(r.get_forecast(12, exog=None if e is None else e.loc[idx]).predicted_mean), scale[idx]))
    e = regressors(TRAIN_END + "-01")
    res = fit(y_[:TRAIN_END], e, spec)
    rebound = []
    for o in ORIGINS:
        r = res.apply(y_[:o], exog=None if e is None else e.loc[y_[:o].index])
        idx = pd.date_range(o + pd.DateOffset(months=1), periods=12, freq="MS")
        rebound.append(mape(np.exp(r.get_forecast(12, exog=None if e is None else e.loc[idx]).predicted_mean), scale[idx]))
    out = dict(name=name, normal=np.mean(normal), rebound=np.mean(rebound), aic=res.aic, converged=bool(res.mle_retvals["converged"]))
    if e is not None:
        explained = 0.0
        for c in e.columns:
            out[f"b_{c}"], out[f"se_{c}"] = res.params[c], res.bse[c]
            if c != "covid":
                share = res.params[c] * change(e[c]) / dy * 100
                out[f"share_{c}"] = share
                explained += share
        out["left"] = 100 - explained
    return out


def table(jobs, workers=16):
    with ProcessPoolExecutor(workers) as ex:
        return pd.DataFrame(list(ex.map(evaluate, jobs)))


def experiment_a():
    combos = {"weekdays only": ["weekdays"], "newlywed + women (current)": ["weekdays", "newlywed", "women"], "rate + lag_pop + women": ["weekdays", "rate", "lag_pop", "women"],
              "rate + women": ["weekdays", "rate", "women"], "rate only": ["weekdays", "rate"], "lag_pop + women (population only)": ["weekdays", "lag_pop", "women"], "women only": ["weekdays", "women"], "newlywed only": ["weekdays", "newlywed"]}
    T = table([(f"{o} | {k}", s, X[c], "2005", False) for o, s in ORDERS.items() for k, c in combos.items()])
    T.to_csv(REP / "08_a_rate_vs_population.csv", index=False)
    print(f"A0 mean log change 2023 -> L12: births {dy:.1f}, newlywed {change(X.newlywed):.1f}, rate {change(X.rate):.1f}, lag_pop {change(X.lag_pop):.1f}, women {change(X.women):.1f}")
    cols = ["name", "normal", "rebound", "aic", "b_rate", "se_rate", "b_lag_pop", "se_lag_pop", "b_newlywed", "se_newlywed", "b_women", "se_women", "share_rate", "share_lag_pop", "share_newlywed", "share_women", "left"]
    print("A1\n", T[cols].round(2).to_string(index=False))


def experiment_b():
    wins = [(0, 11), (3, 14), (6, 17), (9, 20), (12, 23), (24, 35), (0, 23), (3, 26), (6, 29), (9, 32), (12, 35), (18, 41), (24, 47), (0, 35), (12, 47)]
    jobs = []
    for o, s in ORDERS.items():
        for lo, hi in wins:
            ex = pd.DataFrame({"marr": window(lo, hi), "women": X.women, "weekdays": X.weekdays}).loc[lambda f: f.index <= REBOUND_END].dropna()
            jobs.append((f"{o} | {lo}~{hi}", s, ex, "2005", False))
    T = table(jobs)
    T["kind"] = ["explanatory" if int(n.split("| ")[1].split("~")[0]) < 12 else "forecastable" for n in T.name]
    T.to_csv(REP / "08_b_lag_windows.csv", index=False)
    print("B1 windows (explanatory = uses actual future marriages; AIC through 2023-12)\n", T[["name", "kind", "aic", "normal", "rebound", "b_marr", "se_marr", "b_women", "share_marr", "share_women", "left"]].round(2).to_string(index=False))
    for o in ORDERS:
        g = T[T.name.str.startswith(o)]
        print(f"B2 {o} lowest AIC window: {g.loc[g.aic.idxmin(), 'name']} | among forecastable: {g[g.kind == 'forecastable'].loc[g[g.kind == 'forecastable'].aic.idxmin(), 'name']}")


def experiment_c():
    ds = y - y.groupby(y.index.month).transform("mean")
    Xc = X.loc[y.index, ["newlywed", "women"]]
    rows = []
    for end in [f"{yr}-12" for yr in range(2013, 2024)] + ["2026-07"]:
        t, p, _ = coint(ds[:end], Xc[:end], trend="c")
        rows.append(dict(end=end, t=t, p=p))
    EG = pd.DataFrame(rows)
    EG.to_csv(REP / "08_c_engle_granger.csv", index=False)
    print("C1 Engle-Granger by end date\n", EG.round(3).T.to_string(header=False))

    ex = X[["weekdays", "newlywed", "women"]]
    rows = []
    spans = [(f"{a}-01", f"{a + 9}-12") for a in range(2005, 2015)] + [("2016-01", "2026-07")]
    for a, b in spans:
        r = fit(y[a:b], ex, D0)
        rows.append(dict(span=f"{a[:4]}~{b[:7]}", b_newlywed=r.params["newlywed"], se_newlywed=r.bse["newlywed"], b_women=r.params["women"], se_women=r.bse["women"], converged=bool(r.mle_retvals["converged"])))
    R = pd.DataFrame(rows)
    R.to_csv(REP / "08_c_rolling_coefficients.csv", index=False)
    print("C2 d=0 coefficients on 10-year windows\n", R.round(2).to_string(index=False))

    def sup_f(end):
        d = pd.DataFrame({"y": y.diff(12), "nw": X.newlywed.diff(12), "w": X.women.diff(12), "wd": X.weekdays.diff(12)}).reindex(y.index)
        d["ylag"] = d.y.shift(1)
        d = d["2006-02":end].dropna()
        base = sm.OLS(d.y, sm.add_constant(d[["nw", "w", "wd", "ylag"]])).fit()
        n, trim = len(d), int(0.15 * len(d))
        best = (0.0, None)
        for i in range(trim, n - trim):
            brk = (np.arange(n) >= i).astype(float)
            Z = sm.add_constant(d[["nw", "w", "wd", "ylag"]]).assign(nw_b=d.nw * brk, w_b=d.w * brk, b=brk)
            full = sm.OLS(d.y, Z).fit()
            f = (base.ssr - full.ssr) / (full.ssr / full.df_resid)  # Wald form, 3 restrictions: two slope changes and an intercept shift
            if f > best[0]:
                best = (f, d.index[i], full.params[["nw", "w"]].values, full.bse[["nw", "w"]].values, full.params[["nw_b", "w_b"]].values, full.bse[["nw_b", "w_b"]].values)
        return best, n

    for end in ("2023-12", "2026-07"):
        (f, when, pre, pre_se, delta, delta_se), n = sup_f(end)
        print(f"C3 sup-F break, 2006~{end}, n {n}: {f:.2f} at {when.strftime('%Y-%m')} (5% critical value 14.15, Andrews 1993, p=3, 15% trim)")
        print(f"   newlywed slope before {pre[0]:+.2f} (se {pre_se[0]:.2f}), change {delta[0]:+.2f} (se {delta_se[0]:.2f}) | women slope before {pre[1]:+.2f} (se {pre_se[1]:.2f}), change {delta[1]:+.2f} (se {delta_se[1]:.2f})")


AR_PHI, AR_C, AR_RES, MARR_DIFF = None, None, None, None


def placebo_job(args):
    seed, lo, hi = args
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2002-01-01")
    lm = np.log(m.marr.copy())
    d12 = float(MARR_DIFF.loc[start - pd.DateOffset(months=1)])
    for t in lm[start:].index:
        d12 = AR_C + AR_PHI * d12 + rng.choice(AR_RES)
        lm[t] = lm[t - pd.DateOffset(months=12)] + d12
    fake = window(lo, hi, np.exp(lm)).reindex(y.index)
    ex = pd.DataFrame({"weekdays": X.weekdays, "newlywed": fake, "women": X.women}).reindex(y.index)
    res = fit(y[:TRAIN_END], ex, D0)
    scores = []
    for o in ORIGINS:
        r = res.apply(y[:o], exog=ex.loc[y[:o].index])
        idx = pd.date_range(o + pd.DateOffset(months=1), periods=12, freq="MS")
        scores.append(mape(np.exp(r.get_forecast(12, exog=ex.loc[idx]).predicted_mean), level[idx]))
    return dict(seed=seed, rebound=float(np.mean(scores)), b=res.params["newlywed"], t=res.params["newlywed"] / res.bse["newlywed"], rise=change(fake), converged=bool(res.mle_retvals["converged"]))


def experiment_d(reps=1000):
    global AR_PHI, AR_C, AR_RES, MARR_DIFF
    MARR_DIFF = np.log(m.marr).diff(12)
    d = pd.DataFrame({"y": MARR_DIFF, "lag": MARR_DIFF.shift(1)})["2003":"2023"].dropna()
    ols = sm.OLS(d.y, sm.add_constant(d.lag)).fit()
    AR_C, AR_PHI, AR_RES = float(ols.params["const"]), float(ols.params["lag"]), ols.resid.values
    print(f"D0 marriage D12 AR(1) on 2003~2023: c {AR_C:.4f}, phi {AR_PHI:.3f}, resid sd {AR_RES.std():.3f}")
    for lo, hi in ((12, 35), (9, 20)):
        real_win = window(lo, hi).reindex(y.index)
        real_ex = pd.DataFrame({"weekdays": X.weekdays, "newlywed": real_win, "women": X.women}).reindex(y.index)
        real = evaluate(("real", D0, real_ex, "2005", False))
        base = evaluate(("women only", D0, X[["weekdays", "women"]], "2005", False))["rebound"]
        t_real = real["b_newlywed"] / real["se_newlywed"]
        with ProcessPoolExecutor(16) as ex:  # fork inherits the AR globals set above
            D = pd.DataFrame(list(ex.map(placebo_job, [(i, lo, hi) for i in range(reps)], chunksize=8)))
        D.to_csv(REP / f"08_d_placebo_{lo}_{hi}.csv", index=False)
        r_ = real["rebound"]
        print(f"D1 window {lo}~{hi}: real rebound {r_:.2f}, t {t_real:.1f}, rise {change(real_win):.1f} | women only {base:.2f} | reps {len(D)}, not converged {int((~D.converged).sum())}")
        print(f"   placebo rebound median {D.rebound.median():.2f}, 5th pct {D.rebound.quantile(.05):.2f}, min {D.rebound.min():.2f}")
        print(f"   MAPE <= real: {(D.rebound <= r_).mean() * 100:.1f}% | beats women-only: {(D.rebound < base).mean() * 100:.1f}%")
        sig = D[D.t >= 2]
        print(f"   t >= 2 at 2023-12: {len(sig) / len(D) * 100:.1f}%; of those, MAPE <= real: {(sig.rebound <= r_).mean() * 100:.1f}%")
        print(f"   t >= real t: {(D.t >= t_real).mean() * 100:.1f}% | t >= real t and MAPE <= real: {((D.t >= t_real) & (D.rebound <= r_)).mean() * 100:.1f}%")


def experiment_e():
    cols = ["weekdays", "newlywed", "women"]
    exb = X[cols]
    jobs = []
    for o, s in ORDERS.items():
        for start in ("2000", "2005", "2010"):
            jobs.append((f"{o} | start {start} SARIMAX", s, exb, start, False))
            jobs.append((f"{o} | start {start} SARIMA", s, X[["weekdays"]], start, False))
        jobs.append((f"{o} | 2005 + COVID SARIMAX", s, exb, "2005", True))
        jobs.append((f"{o} | 2005 + COVID SARIMA", s, X[["weekdays"]], "2005", True))
    T = table(jobs)
    T.to_csv(REP / "08_e_sensitivity.csv", index=False)
    print("E1 sensitivity\n", T[["name", "normal", "rebound", "b_newlywed", "se_newlywed", "b_women", "se_women", "b_covid", "se_covid", "left", "converged"]].round(2).to_string(index=False))


def experiment_f():
    cols = ["weekdays", "newlywed", "women"]
    ex = X[cols]
    out = {"d=1": {}, "d=0": {}}
    for o, s in ORDERS.items():
        for yr in range(2016, 2024):
            r = fit(y[:f"{yr - 1}-12"], ex, s)
            idx = y[str(yr)].index
            out[o][("normal", yr)] = r.get_forecast(12, exog=ex.loc[idx]).predicted_mean
        res = fit(y[:TRAIN_END], ex, s)
        for t in ORIGINS:
            r = res.apply(y[:t], exog=ex.loc[y[:t].index])
            idx = pd.date_range(t + pd.DateOffset(months=1), periods=12, freq="MS")
            out[o][("rebound", t)] = r.get_forecast(12, exog=ex.loc[idx]).predicted_mean
    rows = []
    for name, w in {"d=1": 1.0, "d=0": 0.0, "50:50": 0.5}.items():
        for period in ("normal", "rebound"):
            scores = []
            for key, f1 in out["d=1"].items():
                if key[0] != period:
                    continue
                f = np.exp(w * f1 + (1 - w) * out["d=0"][key])
                scores.append(mape(f, level[f.index]))
            rows.append(dict(model=name, period=period, mape=np.mean(scores)))
    T = pd.DataFrame(rows).pivot(index="model", columns="period", values="mape").round(2)
    T.to_csv(REP / "08_f_combination.csv")
    main, alt = LOCK["best_each_d"]["1"], LOCK["best_each_d"]["0"]
    assert abs(T.loc["d=1", "rebound"] - summary(main["model"], main["track"], "SARIMAX", "mape")) < 0.02 and abs(T.loc["d=0", "rebound"] - summary(alt["model"], alt["track"], "SARIMAX", "mape")) < 0.02, T
    print("F1 combination; single models reproduce step 3\n", T.to_string())


def path_score_job(args):
    """Mean MAPE over horizons 1~9 only, so a 9-month lag needs no future marriages."""
    name, spec, ex = args
    normal, rebound = [], []
    for yr in range(2016, 2024):
        o = f"{yr - 1}-12-01"
        r = fit(y[:o], ex, spec)
        idx = pd.date_range(pd.Timestamp(o) + pd.DateOffset(months=1), periods=9, freq="MS")
        normal.append(mape(np.exp(r.get_forecast(9, exog=ex.loc[idx]).predicted_mean), level[idx]))
    res = fit(y[:TRAIN_END], ex, spec)
    for o in ORIGINS:
        r = res.apply(y[:o], exog=ex.loc[y[:o].index])
        idx = pd.date_range(o + pd.DateOffset(months=1), periods=9, freq="MS")
        rebound.append(mape(np.exp(r.get_forecast(9, exog=ex.loc[idx]).predicted_mean), level[idx]))
    out = dict(name=name, normal_h9=np.mean(normal), rebound_h9=np.mean(rebound), aic=res.aic)
    if "marr" in ex:
        out.update(b_marr=res.params["marr"], se_marr=res.bse["marr"], b_women=res.params["women"], share_marr=res.params["marr"] * change(ex.marr) / dy * 100, share_women=res.params["women"] * change(ex.women) / dy * 100)
        out["left"] = 100 - out["share_marr"] - out["share_women"]
    return out


def experiment_g():
    jobs = []
    for o, s in ORDERS.items():
        jobs.append((f"{o} | weekdays only", s, X[["weekdays"]].loc[lambda f: f.index <= REBOUND_END]))
        for lo, hi in [(9, 20), (9, 32), (12, 23), (12, 35)]:
            jobs.append((f"{o} | {lo}~{hi}", s, pd.DataFrame({"weekdays": X.weekdays, "marr": window(lo, hi), "women": X.women}).loc[lambda f: f.index <= REBOUND_END].dropna()))
    with ProcessPoolExecutor(6) as ex:
        T = pd.DataFrame(list(ex.map(path_score_job, jobs)))
    level_ = level.copy()
    nv = lambda ts: np.mean([mape(level_.shift(12).loc[pd.date_range(t + pd.DateOffset(months=1), periods=9, freq="MS")], level_.loc[pd.date_range(t + pd.DateOffset(months=1), periods=9, freq="MS")]) for t in ts])
    print(f"G0 seasonal naive h=1~9: normal {nv(pd.to_datetime([f'{yr - 1}-12-01' for yr in range(2016, 2024)])):.2f}, rebound {nv(ORIGINS):.2f}")
    T.to_csv(REP / "08_g_h9.csv", index=False)
    print("G1 real forecasts to 9 months ahead\n", T.round(2).to_string(index=False))


def experiment_h():
    tr = y[:TRAIN_END]
    win = {"lead 1~12": window(-12, -1), "lag 0~8": window(0, 8), "lag 9~20": window(9, 20), "lag 12~35": window(12, 35)}
    rows = []
    for o, s in ORDERS.items():
        base = fit(tr, X[["weekdays", "women"]], s)
        rows.append(dict(order=o, model="weekdays + women", aic=base.aic))
        for name, w in win.items():
            ex = pd.DataFrame({"weekdays": X.weekdays, "women": X.women, "marr": w}).reindex(tr.index)
            assert ex.notna().all().all(), name
            r = fit(tr, ex, s)
            rows.append(dict(order=o, model=name, aic=r.aic, b=r.params["marr"], t=r.params["marr"] / r.bse["marr"]))
        for names in (("lag 0~8", "lag 9~20"), ("lead 1~12", "lag 9~20"), ("lead 1~12", "lag 0~8", "lag 9~20")):
            ex = pd.DataFrame({"weekdays": X.weekdays, "women": X.women, **{n: win[n] for n in names}}).reindex(tr.index)
            r = fit(tr, ex, s)
            rows.append(dict(order=o, model=" + ".join(names), aic=r.aic, **{f"t[{n}]": r.params[n] / r.bse[n] for n in names}, **{f"b[{n}]": r.params[n] for n in names}))
    T = pd.DataFrame(rows)
    T.to_csv(REP / "08_h_lead_lag.csv", index=False)
    print("H1 fit through 2023-12\n", T.round(2).to_string(index=False))


def multistart_job(args):
    key, y_end, spec, ex = args
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    y_ = y[:y_end]
    mod = SARIMAX(y_, exog=None if ex is None else ex.loc[y_.index], **spec)
    single = fit(y_, ex, spec)
    best = single
    rng = np.random.default_rng(zlib.crc32(f"{key}{y_end}".encode()))
    start = np.asarray(single.params)
    for i in range(6):
        sp = np.asarray(mod.start_params) + rng.normal(0, 0.2, len(start)) * (np.abs(np.asarray(mod.start_params)) < 1.5)
        try:
            r = mod.fit(sp, disp=False, maxiter=500)
        except Exception:
            continue
        if r.mle_retvals["converged"] and r.llf > best.llf:
            best = r
    return dict(key=key, y_end=y_end, llf_single=single.llf, llf_best=best.llf, single=single, best=best)


def experiment_i():
    import json
    cands = json.loads((REP / "01_candidates.json").read_text())
    jobs, meta = [], []
    for c in cands:
        spec = dict(order=tuple(c["spec"]["order"]), seasonal_order=tuple(c["spec"]["seasonal_order"]), trend=c["spec"]["trend"])
        cols0 = ["weekdays"] if c["track"] == "weekdays" else []
        for kind, cols in (("SARIMA", cols0), ("SARIMAX", cols0 + ["newlywed", "women"])):
            ex = X[cols] if cols else None
            key = f"{c['model']}|{c['track']}|{kind}"
            for end in [f"{yr}-12-01" for yr in range(2015, 2024)]:
                jobs.append((key, end, spec, ex))
    with ProcessPoolExecutor(8) as pool:
        R = list(pool.map(multistart_job, jobs, chunksize=2))
    rows = []
    for r in R:
        end = pd.Timestamp(r["y_end"])
        ex = None
        spec_key = r["key"]
        track = spec_key.split("|")[1]
        kind = spec_key.split("|")[2]
        cols = (["weekdays"] if track == "weekdays" else []) + (["newlywed", "women"] if kind == "SARIMAX" else [])
        ex = X[cols] if cols else None
        for label_, res in (("single", r["single"]), ("best", r["best"])):
            if end.year == 2023:
                continue
            idx = y[str(end.year + 1)].index
            f = np.exp(res.get_forecast(12, exog=None if ex is None else ex.loc[idx]).predicted_mean)
            rows.append(dict(key=spec_key, fit=label_, target=end.year + 1, mape=mape(f, level[idx]), rmse=np.sqrt(((f - level[idx]) ** 2).mean())))
    M = pd.DataFrame(rows).groupby(["key", "fit"], sort=False)[["mape", "rmse"]].mean().unstack("fit")
    G = pd.DataFrame([dict(key=r["key"], end=r["y_end"][:4], gain=r["llf_best"] - r["llf_single"]) for r in R])
    M["max_llf_gain"] = G.groupby("key").gain.max()
    M["origins_gain_gt_0.01"] = G.assign(flag=lambda f: f.gain > 0.01).groupby("key").flag.sum()
    M.columns = ["_".join(str(x) for x in c if x) if isinstance(c, tuple) else c for c in M.columns]
    M.to_csv(REP / "08_i_multistart.csv")
    print("I1 normal-period MAPE, single start vs best of multi-start\n", M.round(3).to_string())

    ok = {f"{c['model']}|{c['track']}|SARIMAX": c["sarimax_lb_ok"] for c in cands}
    pick = M.reset_index()
    pick = pick[pick.key.isin([k for k, v in ok.items() if v])].sort_values("mape_best")
    near = pick[pick.mape_best < pick.mape_best.iloc[0] + 0.1].sort_values("rmse_best")
    print("I2 declared rule re-applied to multi-start normal-period numbers\n", pick[["key", "mape_single", "mape_best", "rmse_best"]].round(3).to_string(index=False))
    winner = near.iloc[0].key
    print("I3 main model under the rule:", winner)

    # rebound with frozen best-of-multi-start coefficients, every candidate, report only
    reb = []
    for r in R:
        if not r["y_end"].startswith("2023"):
            continue
        track, kind = r["key"].split("|")[1:]
        cols = (["weekdays"] if track == "weekdays" else []) + (["newlywed", "women"] if kind == "SARIMAX" else [])
        ex = X[cols] if cols else None
        res = r["best"]
        scores = []
        for o in ORIGINS:
            a = res.apply(y[:o], exog=None if ex is None else ex.loc[y[:o].index])
            idx = pd.date_range(o + pd.DateOffset(months=1), periods=12, freq="MS")
            scores.append(mape(np.exp(a.get_forecast(12, exog=None if ex is None else ex.loc[idx]).predicted_mean), level[idx]))
        reb.append(dict(key=r["key"], rebound_best=np.mean(scores), llf_gain_2023=r["llf_best"] - r["llf_single"]))
    RB = pd.DataFrame(reb)
    RB.to_csv(REP / "08_i_multistart_rebound.csv", index=False)
    print("I4 rebound MAPE with multi-start coefficients\n", RB.round(3).to_string(index=False))


def experiment_j():
    jobs = []
    for lo, hi in ((9, 20), (9, 32)):
        pop = np.log(sum(w2534.shift(k) for k in range(lo, hi + 1)) / (hi - lo + 1)).reindex(idx_x)
        win = window(lo, hi).reindex(idx_x)
        F = pd.DataFrame({"weekdays": X.weekdays, "marr": win, "rate": win - pop, "lag_pop": pop, "women": X.women}).loc[lambda f: f.index <= REBOUND_END]
        print(f"J0 window {lo}~{hi} mean log change 2023 -> L12: marriages {change(F.marr):.1f} = rate {change(F.rate):.1f} + population {change(F.lag_pop):.1f}; births {dy:.1f}, women 30~34 {change(F.women):.1f}")
        for o, s_ in ORDERS.items():
            for nm, cols in {"marr + women": ["weekdays", "marr", "women"], "rate + lag_pop + women": ["weekdays", "rate", "lag_pop", "women"], "rate + women": ["weekdays", "rate", "women"], "lag_pop + women": ["weekdays", "lag_pop", "women"]}.items():
                jobs.append((f"{o} | {lo}~{hi} | {nm}", s_, F[cols], "2005", False))
    T = table(jobs)
    T.to_csv(REP / "08_j_rate_vs_population_short_lag.csv", index=False)
    print("J1 explanatory, actual marriages\n", T[["name", "aic", "normal", "rebound", "b_marr", "se_marr", "b_rate", "se_rate", "b_lag_pop", "se_lag_pop", "b_women", "se_women", "share_marr", "share_rate", "share_lag_pop", "share_women", "left"]].round(2).to_string(index=False))


def k_job(args):
    name, spec, lo, hi = args
    if lo is None:
        hist = X[["weekdays", "women"]].loc[lambda f: f.index <= REBOUND_END]
    else:
        hist = pd.DataFrame({"weekdays": X.weekdays, "marr": window(lo, hi), "women": X.women}).loc[lambda f: f.index <= REBOUND_END].dropna()

    def future(o, idx):
        out = hist.loc[idx].copy()
        if lo is None:
            return out
        ext = m.marr[:o].copy()
        for t in pd.date_range(o + pd.DateOffset(months=1), o + pd.DateOffset(months=12), freq="MS"):
            ext[t] = ext[t - pd.DateOffset(months=12)]  # same month last year: no foresight
        out["marr"] = window(lo, hi, ext).reindex(idx)
        assert out.notna().all().all()
        return out

    normal, rebound = [], []
    for yr in range(2016, 2024):
        o = pd.Timestamp(f"{yr - 1}-12-01")
        r = fit(y[:o], hist, spec)
        idx = pd.date_range(o + pd.DateOffset(months=1), periods=12, freq="MS")
        normal.append(mape(np.exp(r.get_forecast(12, exog=future(o, idx)).predicted_mean), level[idx]))
    res = fit(y[:TRAIN_END], hist, spec)
    for o in ORIGINS:
        r = res.apply(y[:o], exog=hist.loc[y[:o].index])
        idx = pd.date_range(o + pd.DateOffset(months=1), periods=12, freq="MS")
        rebound.append(mape(np.exp(r.get_forecast(12, exog=future(o, idx)).predicted_mean), level[idx]))
    out = dict(name=name, normal=np.mean(normal), rebound=np.mean(rebound), aic=res.aic)
    if lo is not None:
        out.update(b_marr=res.params["marr"], se_marr=res.bse["marr"], b_women=res.params["women"], se_women=res.bse["women"])
    return out


def experiment_k():
    jobs = []
    for o, s_ in ORDERS.items():
        jobs.append((f"{o} | women only", s_, None, None))
        for lo, hi in ((9, 20), (9, 32), (12, 23), (12, 35)):
            jobs.append((f"{o} | {lo}~{hi}", s_, lo, hi))
    with ProcessPoolExecutor(8) as ex:
        T = pd.DataFrame(list(ex.map(k_job, jobs)))
    T.to_csv(REP / "08_k_official_metric.csv", index=False)
    main, alt = LOCK["best_each_d"]["1"], LOCK["best_each_d"]["0"]
    for o, v in (("d=1", main), ("d=0", alt)):
        got = T[T.name == f"{o} | 12~35"].iloc[0]
        assert abs(got.rebound - summary(v["model"], v["track"], "SARIMAX", "mape")) < 0.02 and abs(got.normal - summary(v["model"], v["track"], "SARIMAX", "normal_mape")) < 0.02, got
    print(f"K0 12~35 reproduces step 2 and 3; seasonal naive normal {summary('seasonal naive', '-', 'baseline', 'normal_mape'):.2f}, rebound {summary('seasonal naive', '-', 'baseline', 'mape'):.2f}")
    print("K1 official metric, marriages after each origin filled with same month last year\n", T.round(2).to_string(index=False))


if __name__ == "__main__":
    print("cpu", os.cpu_count())
    which = sys.argv[1] if len(sys.argv) > 1 else "acefbd"
    steps = {"a": experiment_a, "b": experiment_b, "c": experiment_c, "d": experiment_d, "e": experiment_e, "f": experiment_f, "g": experiment_g, "h": experiment_h, "i": experiment_i, "j": experiment_j, "k": experiment_k}
    for k in which:
        print(f"\n===== {k.upper()} =====", flush=True)
        steps[k]()
        sys.stdout.flush()
