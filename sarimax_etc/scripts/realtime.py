"""Real-time evaluation with publication lag, vintage proxy, equal-length period comparison, MASE, DM, distributed lag, scenarios.

Report only; rules are in notes/decisions.md.
"""
import json
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))  # appended last: scripts run with python -I
from common import RAW, REBOUND_END, REP, fit, load_monthly, weekdays, women_30_34  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

warnings.filterwarnings("ignore")
pd.set_option("display.width", 190)
pd.set_option("display.max_columns", 40)
PUB_LAG = 2
mo = lambda n: pd.DateOffset(months=n)
m = load_monthly()
y = np.log(m.births["2005":REBOUND_END]).rename("y")
level = np.exp(y)
log_w = np.log(women_30_34())
log_wd = np.log(weekdays(pd.date_range("2004-01-01", "2032-12-01", freq="MS")))
LOCK = json.loads((REP / "02_locked.json").read_text())
SPECS = {(c["track"], c["model"]): c["spec"] for c in json.loads((REP / "01_candidates.json").read_text())}


def spec_of(d):
    v = LOCK["best_each_d"][d]
    s = SPECS[(v["track"], v["model"])]
    return dict(order=tuple(s["order"]), seasonal_order=tuple(s["seasonal_order"]), trend=s["trend"])


D1, D0 = spec_of("1"), spec_of("0")
assert D1["order"] == (0, 1, 1) and D0["order"] == (1, 0, 0), "orders changed; review before reading results"
NORMAL = pd.to_datetime([f"{yr}-12-01" for yr in range(2015, 2023)])
BLOCKS = {"2015-12 start": "2015-12-01", "2017-08 start": "2017-08-01", "2019-04 start": "2019-04-01", "2020-12 start": "2020-12-01", "rebound 2023-12": "2023-12-01"}
VINTAGE = pd.Timestamp("2023-12-01")  # the team's projection file was published in December 2023


def extend_marr(known_to, end):
    """Marriages published by known_to, then the same month a year earlier: no foresight."""
    out = m.marr[:known_to].copy()
    for t in pd.date_range(known_to + mo(1), end, freq="MS"):
        out[t] = out[t - mo(12)]
    return out


def window(lo, hi, src):
    idx = pd.date_range(src.index[0], src.index[-1] + mo(lo), freq="MS")
    full = src.reindex(idx)
    return np.log(sum(full.shift(k) for k in range(lo, hi + 1)) * 12 / (hi - lo + 1))


def almon(lo, hi, deg, src):
    """Quadratic-weight distributed lag on log monthly marriages; z_j is normalized by the window length."""
    idx = pd.date_range(src.index[0], src.index[-1] + mo(lo), freq="MS")
    logm = np.log(src).reindex(idx)
    n = hi - lo + 1
    return {f"z{j}": sum((((k - lo) / (hi - lo)) ** j) * logm.shift(k) for k in range(lo, hi + 1)) / n for j in range(deg + 1)}


def women_series(o, known_to, idx, mode):
    if mode == "file" or o >= VINTAGE:
        return log_w.reindex(idx)
    slope = (log_w[known_to] - log_w[known_to - mo(12)]) / 12  # projection vintage of the time is not in the team's file
    out = log_w.reindex(idx).copy()
    future = idx[idx > known_to]
    out[future] = log_w[known_to] + slope * np.arange(1, len(future) + 1)
    return out


def frames(model, o, months=12):
    known_to = o - mo(PUB_LAG)
    end = o + mo(months)
    idx = pd.date_range(y.index[0], end, freq="MS")
    cols = {"weekdays": log_wd.reindex(idx)} if model.get("calendar", True) else {}
    if model.get("women"):
        cols["women"] = women_series(o, known_to, idx, model["women"])
    kind = model.get("kind")
    if kind == "win":
        cols["marr"] = window(model["lo"], model["hi"], extend_marr(known_to, end)).reindex(idx)
    elif kind == "almon":
        for k, v in almon(model["lo"], model["hi"], 2, extend_marr(known_to, end)).items():
            cols[k] = v.reindex(idx)
    elif kind == "bins":
        ext = extend_marr(known_to, end)
        for a in range(0, 48, 6):
            cols[f"lag{a}"] = window(a, a + 5, ext).reindex(idx)
    ex = pd.DataFrame(cols, index=idx)
    assert ex.notna().all().all(), (model["name"], o)
    return ex, known_to


def xo(part):
    return part if part.shape[1] else None  # a pure SARIMA has no regressors


def path(res_or_none, model, o, res0=None, months=12):
    ex, known_to = frames(model, o, months)
    tr = y[:known_to]
    r = fit(tr, xo(ex), model["spec"]) if res0 is None else res0.apply(tr, exog=xo(ex.loc[tr.index]))
    fut = ex.loc[known_to + mo(1):]
    f = np.exp(r.get_forecast(len(fut), exog=xo(fut)).predicted_mean)
    tg = pd.date_range(o + mo(1), periods=months, freq="MS")
    return r, f[tg]


def naive_path(o):
    known_to = o - mo(PUB_LAG)
    tg = pd.date_range(o + mo(1), periods=12, freq="MS")
    return pd.Series([level[t - mo(12)] if t - mo(12) <= known_to else level[t - mo(24)] for t in tg], index=tg)


def mase_scale(known_to):
    s = level[:known_to]
    return float((s - s.shift(12)).abs().mean())


def metrics(f, scale):
    a = level[f.index]
    return dict(mape=float((np.abs(f - a) / a).mean() * 100), mase=float(np.abs(f - a).mean() / scale), rmse=float(np.sqrt(((f - a) ** 2).mean())), bias=float(((f / a - 1) * 100).mean()), ape12=float(abs(f.iloc[-1] / a.iloc[-1] - 1) * 100))


def evaluate(model):
    """Official normal (8 December decisions, re-estimated) plus five equal blocks of 20 frozen monthly decisions."""
    rows = []
    for o in NORMAL:
        _, f = path(None, model, o)
        rows.append(dict(period="official normal", o=o, **metrics(f, mase_scale(o - mo(PUB_LAG)))))
    aic = None
    for label, start in BLOCKS.items():
        o0 = pd.Timestamp(start)
        scale = mase_scale(o0 - mo(PUB_LAG))
        res0 = None
        for o in pd.date_range(o0, periods=20, freq="MS"):
            if res0 is None:
                res0, f = path(None, model, o)
                if label.startswith("rebound"):
                    aic = res0.aic
                    params = res0.params
                    bse = res0.bse
                    assert res0.mle_retvals["converged"], model["name"]
            else:
                _, f = path(None, model, o, res0=res0)
            rows.append(dict(period=label, o=o, **metrics(f, scale)))
    R = pd.DataFrame(rows).assign(name=model["name"])
    info = dict(name=model["name"], aic=aic, **{f"b_{c}": params[c] for c in params.index if c in ("marr", "women", "z0", "z1", "z2")}, **{f"se_{c}": bse[c] for c in bse.index if c in ("marr", "women")})
    return R, info


def naive_rows():
    rows = []
    for o in NORMAL:
        rows.append(dict(period="official normal", o=o, **metrics(naive_path(o), mase_scale(o - mo(PUB_LAG)))))
    for label, start in BLOCKS.items():
        o0 = pd.Timestamp(start)
        scale = mase_scale(o0 - mo(PUB_LAG))
        for o in pd.date_range(o0, periods=20, freq="MS"):
            rows.append(dict(period=label, o=o, **metrics(naive_path(o), scale)))
    return pd.DataFrame(rows).assign(name="seasonal naive")


def dm(a, b, h=12):
    """Harvey-Leybourne-Newbold corrected DM on absolute percentage errors; negative means a is better."""
    d = (a - b).values
    n = len(d)
    u = d - d.mean()
    v = (u @ u) / n
    for k in range(1, h):
        v += 2 * (1 - k / h) * (u[k:] @ u[:-k]) / n  # Bartlett weights keep the variance positive
    se = np.sqrt(v / n)
    corr = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    stat = d.mean() / se * corr
    return dict(diff=d.mean(), dm=stat, p=2 * stats.t.sf(abs(stat), n - 1), detectable=2.8 * se / corr)


def block_lags():
    bins = [(k, k + 5) for k in range(0, 48, 6)]
    o = pd.Timestamp("2023-12-01")
    known_to = o - mo(PUB_LAG)
    tr = y[:known_to]
    rows, shapes = [], []
    for d, spec in (("d=1", D1), ("d=0", D0)):
        ex = pd.DataFrame({"weekdays": log_wd, "women": log_w, **{f"lag {a}~{b}": window(a, b, m.marr) for a, b in bins}}).reindex(tr.index)
        r = fit(tr, ex, spec)
        for a, b in bins:
            c = f"lag {a}~{b}"
            rows.append(dict(order=d, lag=f"{a}~{b}", b=r.params[c], se=r.bse[c], t=r.params[c] / r.bse[c]))
        names = [f"lag {a}~{b}" for a, b in bins]
        cov_b = r.cov_params().loc[names, names].values
        shapes.append(dict(order=d, aic=r.aic, total=r.params[names].sum(), se_total=float(np.sqrt(cov_b.sum()))))
        for lo, hi in ((14, 47), (9, 47)):
            ex2 = pd.DataFrame({"weekdays": log_wd, "women": log_w, **almon(lo, hi, 2, m.marr)}).reindex(tr.index)
            r2 = fit(tr, ex2, spec)
            p = [(k - lo) / (hi - lo) for k in range(lo, hi + 1)]
            c = r2.params[["z0", "z1", "z2"]].values
            w = np.array([(c[0] + c[1] * x + c[2] * x * x) / (hi - lo + 1) for x in p])
            grad = np.array([sum(x**j for x in p) / (hi - lo + 1) for j in range(3)])
            cov = r2.cov_params().loc[["z0", "z1", "z2"], ["z0", "z1", "z2"]].values
            shapes.append(dict(order=d, almon=f"{lo}~{hi}", aic=r2.aic, total=w.sum(), se_total=float(np.sqrt(grad @ cov @ grad)), peak_lag=lo + int(np.argmax(w)), weight_at=dict(zip([lo, (lo + hi) // 2, hi], np.round([w[0], w[(hi - lo) // 2], w[-1]], 4).tolist()))))
    return pd.DataFrame(rows), pd.DataFrame(shapes)


def scenarios(lo, hi, women=True, calendar=True):
    """Marriage rate (hold, -5%/year) x population (projected, frozen at 2026-07), refit through 2026-07."""
    p = pd.read_csv(RAW / "성_및_연령별_추계인구_1세별_5세별_전국.csv", header=None, dtype=str)
    years = p.iloc[2, 1:].astype(int).values
    P = p.iloc[3:].set_index(0)
    grp = lambda g: pd.Series(P.loc[g].astype(float).values, index=pd.to_datetime([f"{yr}-07-01" for yr in years]))
    raw = grp("25 - 29세") + grp("30 - 34세")
    pop = raw.reindex(pd.date_range(raw.index[0], raw.index[-1], freq="MS")).interpolate("linear")
    last = m.index[-1]
    end_m, end = pd.Timestamp("2031-12-01"), pd.Timestamp("2031-12-01")
    fut_m = pd.date_range(last + mo(1), end_m, freq="MS")
    rate = (m.marr / pop)[last - mo(11):last]
    out, coef = [], []
    for d, spec in (("d=1", D1), ("d=0", D0)):
        base = None
        for rname, g in (("hold", 0.0), ("decline", -0.05)):
            for pname, frozen in (("projected", False), ("frozen", True)):
                ext = m.marr.copy()
                for i, t in enumerate(fut_m):
                    r_ = rate[rate.index.month == t.month].iloc[0] * (1 + g) ** (i // 12 + 1)
                    ext[t] = r_ * (pop[last] if frozen else pop[t])
                wom = log_w.copy()
                if frozen:
                    wom[wom.index > last] = log_w[last]
                cols = {"weekdays": log_wd} if calendar else {}
                cols |= {"marr": window(lo, hi, ext)} | ({"women": wom} if women else {})
                X = pd.DataFrame(cols).reindex(pd.date_range(y.index[0], end, freq="MS"))
                if base is None:
                    base = fit(y, X, spec)
                    coef.append(dict(order=d, window=f"{lo}~{hi}", b_marr=base.params["marr"], se_marr=base.bse["marr"], b_women=base.params.get("women"), se_women=base.bse.get("women")))
                r = base.apply(y, exog=X.loc[y.index])
                f = np.exp(r.get_forecast(len(fut_m), exog=X.loc[fut_m]).predicted_mean)
                ann = f.groupby(f.index.year).sum()
                ann.loc[2026] += level["2026"].sum()
                for yr, v in ann.items():
                    out.append(dict(order=d, window=f"{lo}~{hi}", rate=rname, population=pname, year=int(yr), births=float(v)))
    return pd.DataFrame(out), pd.DataFrame(coef)


if __name__ == "__main__":
    models = []
    for d, spec in (("d=1", D1), ("d=0", D0)):
        models.append(dict(name=f"{d} | no X", spec=spec))
        models.append(dict(name=f"{d} | women", spec=spec, women="safe"))
        for lo, hi in ((14, 37), (12, 35), (9, 32), (9, 20), (14, 25), (14, 49), (14, 61)):
            models.append(dict(name=f"{d} | {lo}~{hi}", spec=spec, women="safe", kind="win", lo=lo, hi=hi))
        models.append(dict(name=f"{d} | 12~35, women file", spec=spec, women="file", kind="win", lo=12, hi=35))
        for lo, hi in ((14, 47), (9, 47)):
            models.append(dict(name=f"{d} | Almon {lo}~{hi}", spec=spec, women="safe", kind="almon", lo=lo, hi=hi))
        models.append(dict(name=f"{d} | DL bins 0~47", spec=spec, women="safe", kind="bins"))
    cands = json.loads((REP / "01_candidates.json").read_text())
    for c in cands:
        sp = dict(order=tuple(c["spec"]["order"]), seasonal_order=tuple(c["spec"]["seasonal_order"]), trend=c["spec"]["trend"])
        models.append(dict(name=f"cand | {c['track']} | {c['model']}", spec=sp, women="safe", kind="win", lo=12, hi=35, calendar=c["track"] == "weekdays"))
    with ProcessPoolExecutor(16) as ex:
        res = list(ex.map(evaluate, models))
    R = pd.concat([naive_rows()] + [r for r, _ in res], ignore_index=True)
    R.to_csv(REP / "09_realtime_by_origin.csv", index=False)
    info = pd.DataFrame([i for _, i in res])
    info.to_csv(REP / "09_realtime_info.csv", index=False)

    S = R.groupby(["name", "period"], sort=False)[["mape", "mase", "rmse", "bias", "ape12"]].mean().reset_index()
    S.to_csv(REP / "09_realtime_summary.csv", index=False)
    wide = S.pivot_table(index="name", columns="period", values="mape", sort=False)[["official normal", "rebound 2023-12"]].rename(columns={"official normal": "normal", "rebound 2023-12": "rebound"})
    wide["normal_mase"] = S[S.period == "official normal"].set_index("name").mase
    wide["rebound_mase"] = S[S.period == "rebound 2023-12"].set_index("name").mase
    wide["ape12_rebound"] = S[S.period == "rebound 2023-12"].set_index("name").ape12
    T1 = wide.join(info.set_index("name")[["aic", "b_marr", "se_marr", "b_women", "se_women"]])
    print("T1 real-time results; normal = 8 December decisions, rebound = 20 frozen decisions\n", T1.round(2).to_string())

    C = T1[T1.index.str.startswith("cand")].copy()
    C["rmse"] = S[S.period == "official normal"].set_index("name").rmse
    C["lb_ok"] = [next(c["sarimax_lb_ok"] for c in cands if f"cand | {c['track']} | {c['model']}" == n) for n in C.index]
    ok = C[C.lb_ok].sort_values("normal")
    near = ok[ok.normal < ok.normal.iloc[0] + 0.1].sort_values("rmse")
    print("T0 main-model rule on real-time normal numbers, window 12~35\n", C[["normal", "rebound", "rmse", "lb_ok"]].round(2).to_string())
    print("T0 main under the rule:", near.index[0])

    keep = ["seasonal naive", "d=1 | no X", "d=1 | 12~35", "d=1 | 14~37", "d=1 | 9~32", "d=0 | no X", "d=0 | 12~35", "d=0 | 14~37", "d=0 | 9~32"]
    B = S[S.name.isin(keep) & S.period.isin(BLOCKS)]
    print("T2 equal blocks of 20 frozen decisions: MAPE\n", B.pivot_table(index="name", columns="period", values="mape", sort=False).loc[keep, list(BLOCKS)].round(2).to_string())
    print("T3 equal blocks: MASE\n", B.pivot_table(index="name", columns="period", values="mase", sort=False).loc[keep, list(BLOCKS)].round(2).to_string())

    ap = R[R.period == "rebound 2023-12"].pivot_table(index="o", columns="name", values="ape12", sort=False)
    pairs = [("d=1 | 12~35", "seasonal naive"), ("d=1 | 9~32", "seasonal naive"), ("d=0 | 12~35", "seasonal naive"), ("d=0 | 9~32", "seasonal naive"), ("d=1 | 9~32", "d=1 | 12~35"), ("d=1 | 9~32", "d=1 | no X"), ("d=0 | 9~32", "d=0 | no X"), ("d=0 | 9~32", "d=1 | 9~32")]
    D = pd.DataFrame([dict(a=a, b=b, **dm(ap[a], ap[b])) for a, b in pairs])
    D.to_csv(REP / "09_dm_tests.csv", index=False)
    print("T4 DM on 12-month-ahead errors, 20 overlapping decisions; detectable = rough 80%-power difference in points\n", D.round(3).to_string(index=False))

    BL, SH = block_lags()
    BL.to_csv(REP / "09_lag_blocks.csv", index=False)
    print("T5 six-month lag bins, fit through 2023-10\n", BL.round(2).pivot_table(index="lag", columns="order", values=["b", "t"], sort=False).to_string())
    print("T6 totals and Almon shapes\n", SH.round(3).to_string(index=False))

    frames_, coefs = [], []
    for lo, hi in ((9, 32), (12, 35)):
        sc, co = scenarios(lo, hi)
        frames_.append(sc)
        coefs.append(co)
    SC = pd.concat(frames_, ignore_index=True)
    SC.to_csv(REP / "09_scenarios.csv", index=False)
    print("T7 coefficients through 2026-07\n", pd.concat(coefs).round(3).to_string(index=False))
    for (d, w), g in SC.groupby(["order", "window"], sort=False):
        piv = g.pivot_table(index="year", columns=["rate", "population"], values="births", sort=False) / 1000
        print(f"T8 {d} window {w}, annual births (thousands)\n", piv.round(1).to_string())
        y31 = piv.loc[2031]
        base = y31[("hold", "frozen")]
        struct = y31[("hold", "projected")] - base
        behav = y31[("decline", "frozen")] - base
        inter = y31[("decline", "projected")] - base - struct - behav
        print(f"   2031 against hold x frozen ({base:.1f}): structure {struct:+.1f}, behavior {behav:+.1f}, interaction {inter:+.1f}")
