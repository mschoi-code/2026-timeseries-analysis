"""Which SARIMAX spec yields the EDA's 7.1% rebound MAPE, and does that spec hold in the normal period?"""
import itertools
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, coint

warnings.filterwarnings("ignore")
raw = Path(sys.argv[1])
m = pd.read_csv(raw / "월간_인구동향_출생_혼인.csv", skiprows=2, header=None, names=["t", "births", "marr"], dtype={"t": str})
m.index = pd.to_datetime(m.t.str.slice(0, 7), format="%Y.%m")
full = m.reindex(pd.date_range(m.index[0], m.index[-1] + pd.DateOffset(months=12), freq="MS"))
nw = sum(full.marr.shift(k) for k in range(12, 36)) / 2
p = pd.read_csv(raw / "성_및_연령별_추계인구_1세별_5세별_전국.csv", header=None, dtype=str)
yrs = p.iloc[2, 1:].astype(int).values
w = pd.Series(p.iloc[3:].set_index(0).loc["30 - 34세"].astype(float).values, index=yrs)
wl = pd.Series(w.values, index=pd.to_datetime([f"{y}-07-01" for y in yrs]))
wl = wl.reindex(pd.date_range(wl.index[0], wl.index[-1], freq="MS")).interpolate()
ws = pd.Series(w.reindex(wl.index.year).values, index=wl.index)  # annual step
y = np.log(m.births["2005":])
FORMS = {
    "log,lin": pd.DataFrame({"a": np.log(nw), "b": np.log(wl)}),
    "log,step": pd.DataFrame({"a": np.log(nw), "b": np.log(ws)}),
    "lvl,lin": pd.DataFrame({"a": nw / 1e5, "b": wl / 1e6}),
    "lvl,step": pd.DataFrame({"a": nw / 1e5, "b": ws / 1e6}),
}
BASE = {"marr_t": np.log(m.marr).to_frame("a"), "marr_l12": np.log(full.marr).shift(12).to_frame("a")}


def fc(X, end, h, order, sorder, trend):
    tr = y[:end]
    idx = pd.date_range(tr.index[-1] + pd.DateOffset(months=1), periods=h, freq="MS")
    idx = idx[idx <= y.index[-1]]
    ex = None if X is None else X.loc[tr.index]
    fit = SARIMAX(tr, exog=ex, order=order, seasonal_order=sorder, trend=trend).fit(disp=False, maxiter=150)
    f = fit.get_forecast(len(idx), exog=None if X is None else X.loc[idx]).predicted_mean
    return fit, np.exp(f), np.exp(y.loc[idx])


def mape(f, a):
    return float((abs(f - a) / a).mean() * 100)


def job(a):
    form, (pp, d, q), (D, trend) = a
    try:
        order, sorder = (pp, d, q), (0, D, 1, 12) if D else (1, 0, 1, 12)
        fit, f, ac = fc(FORMS[form], "2023-12", 31, order, sorder, trend)
        return dict(form=form, order=order, D=D, trend=trend, mape=mape(f, ac), ba=fit.params["a"], sa=fit.bse["a"], bb=fit.params["b"], sb=fit.bse["b"], y26=float(((f / ac - 1) * 100)["2026"].mean()))
    except Exception as e:
        return dict(form=form, order=(pp, d, q), D=D, trend=trend, mape=np.nan, err=str(e)[:40])


if __name__ == "__main__":
    grid = list(itertools.product(FORMS, [(0, 1, 1), (1, 1, 1), (0, 0, 1), (1, 0, 0), (2, 0, 0), (1, 0, 1), (2, 0, 1)], [(1, "n"), (1, "c"), (0, "n"), (0, "c")]))
    with ProcessPoolExecutor(8) as ex:
        G = pd.DataFrame(list(ex.map(job, grid)))
    G["d"] = G.order.map(lambda o: o[1])
    print("S1 fits", len(G), "failed", int(G.mape.isna().sum()))
    print("S2 best MAPE by (d, D):\n", G.groupby(["d", "D"]).mape.agg(["min", "median", "max"]).round(1).to_string())
    near = G[(G.mape - 7.1).abs() < 1.0].sort_values("mape")
    print("S3 specs within 1pt of 7.1%:\n", near[["form", "order", "D", "trend", "mape", "ba", "sa", "bb", "sb", "y26"]].round(3).to_string())
    print("S4 d=1,D=1 all:\n", G[(G.d == 1) & (G.D == 1)].sort_values("mape")[["form", "order", "trend", "mape", "ba", "sa", "bb", "sb"]].head(8).round(3).to_string())

    # baseline EDA triple: SARIMA 18.0 / raw marriage 17.4 / rolling numbers
    for name, X in {"none": None, **BASE}.items():
        fit, f, ac = fc(X, "2023-12", 31, (0, 1, 1), (0, 1, 1, 12), None)
        print(f"S5 (0,1,1)(0,1,1) X={name}: MAPE {mape(f, ac):.1f}", "" if X is None else f"b {fit.params['a']:+.3f} se {fit.bse['a']:.3f}")

    # rolling 12-step-ahead with monthly re-estimation (EDA: SARIMAX 6.3, Naive 8.7, SARIMA 11.1)
    tgt = pd.date_range("2024-12-01", "2026-07-01", freq="MS")
    def roll(X, order, sorder, trend=None):
        out = {}
        for T in tgt:
            o = T - pd.DateOffset(months=12)
            _, f, ac = fc(X, o, 12, order, sorder, trend)
            out[T] = (f.iloc[-1], ac.iloc[-1])
        d = pd.DataFrame(out, index=["f", "a"]).T
        return mape(d.f, d.a)
    act = np.exp(y)
    print("S6 rolling h=12, 20 origins: seasonal naive %.1f (all 31 targets %.1f) | SARIMA %.1f | SARIMAX d=1 %.1f" % (
        mape(act.shift(12).loc[tgt], act.loc[tgt]),
        mape(act.shift(12).loc["2024-01":"2026-07"], act.loc["2024-01":"2026-07"]),
        roll(None, (0, 1, 1), (0, 1, 1, 12)), roll(FORMS["log,lin"], (0, 1, 1), (0, 1, 1, 12))))

    # the best-looking spec under the proposal's own selection rule: normal-period 8 origins
    cand = near.head(3).to_dict("records")
    rows = []
    for yr in range(2015, 2023):
        e = f"{yr}-12"
        row = {"target": yr + 1, "SARIMA": mape(*fc(None, e, 12, (0, 1, 1), (0, 1, 1, 12), None)[1:])}
        row["d1_X"] = mape(*fc(FORMS["log,lin"], e, 12, (0, 1, 1), (0, 1, 1, 12), None)[1:])
        for i, c in enumerate(cand):
            so = (0, c["D"], 1, 12) if c["D"] else (1, 0, 1, 12)
            fit, f, ac = fc(FORMS[c["form"]], e, 12, c["order"], so, c["trend"])
            row[f"c{i}"] = mape(f, ac)
            row[f"c{i}_ba"] = fit.params["a"]
        rows.append(row)
    R = pd.DataFrame(rows).set_index("target")
    print("S7 normal period with near-7.1 specs:\n", R.round(2).to_string(), "\n means:\n", R.filter(regex="^(SARIMA|d1_X|c[0-9])$").mean().round(2).to_string())

    # is a level relation defensible at all? (d=0 specs assume it)
    ds = y - y.groupby(y.index.month).transform("mean")
    Xc = FORMS["log,lin"]
    for end in ("2015-12", "2023-12", "2026-07"):
        t, pv, _ = coint(ds[:end], Xc.loc[ds[:end].index], trend="c")
        print(f"S8 Engle-Granger deseasonalized log births ~ log nw + log w, 2005~{end}: t {t:.2f} p {pv:.3f}")
    res = {}
    for end in ("2015-12", "2023-12"):
        X_ = np.column_stack([np.ones(len(ds[:end])), Xc.loc[ds[:end].index].values])
        b = np.linalg.lstsq(X_, ds[:end].values, rcond=None)[0]
        u = ds[:end].values - X_ @ b
        res[end] = dict(b_nw=round(b[1], 2), b_w=round(b[2], 2), DW=round(float((np.diff(u) ** 2).sum() / (u ** 2).sum()), 2), R2=round(1 - u.var() / ds[:end].var(), 3))
    print("S9 OLS level regression:", res)
