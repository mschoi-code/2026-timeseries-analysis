"""Shared data, exogenous features, metrics and plot style for the SARIMA/SARIMAX steps."""
import os
import zlib

os.environ.setdefault("OMP_NUM_THREADS", "1")  # must precede numpy: process pools oversubscribe otherwise
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
DATA = ROOT.parent / "data"  # team-shared KOSIS CSVs (births/marriages, population projection)
FIG = ROOT / "figures"
REP = ROOT / "reports"
FIG.mkdir(exist_ok=True)
REP.mkdir(exist_ok=True)

START = "2005-01"
SELECT_END = "2015-12"  # order selection sees nothing after this
TRAIN_END = "2023-12"   # rebound-period coefficients are frozen here
REBOUND_END = "2026-07"

S = 12
PROPOSAL = dict(order=(0, 1, 1), seasonal_order=(0, 1, 1, S), trend=None)
EDA_71 = dict(order=(1, 0, 0), seasonal_order=(0, 1, 1, S), trend="c")

INK, MUTED = "#0b0b0b", "#898781"
BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({
    "font.family": "Malgun Gothic", "axes.unicode_minus": False,
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.6,
    "legend.frameon": False, "font.size": 10, "savefig.dpi": 200, "savefig.bbox": "tight",
})


def load_monthly():
    """Monthly births and marriages from the team's KOSIS file, validated."""
    m = pd.read_csv(DATA / "월간_인구동향_출생_혼인.csv", skiprows=2, header=None,
                    names=["t", "births", "marr"], dtype={"t": str})
    m.index = pd.to_datetime(m.t.str.slice(0, 7), format="%Y.%m")
    m = m.drop(columns="t").astype(float)
    assert m.index.equals(pd.date_range(m.index[0], m.index[-1], freq="MS")), "gaps in months"
    assert (m > 0).all().all(), "missing or non-positive values"
    return m.asfreq("MS")


def newlywed(marr):
    """Marriages 12~35 months earlier / 2; known 12 months past the last observed marriage."""
    idx = pd.date_range(marr.index[0], marr.index[-1] + pd.DateOffset(months=12), freq="MS")
    full = marr.reindex(idx)
    return sum(full.shift(k) for k in range(12, 36)) / 2


def women_30_34():
    """Women aged 30~34 by month: annual values sit on July 1 (mid-year population), linear between."""
    p = pd.read_csv(DATA / "성_및_연령별_추계인구_1세별_5세별_전국.csv", header=None, dtype=str)
    years = p.iloc[2, 1:].astype(int).values
    w = pd.Series(p.iloc[3:].set_index(0).loc["30 - 34세"].astype(float).values,
                  index=pd.to_datetime([f"{y}-07-01" for y in years]))
    return w.reindex(pd.date_range(w.index[0], w.index[-1], freq="MS")).interpolate("linear")


def weekdays(index):
    """Mon~Fri days per month; Seollal/Chuseok-adjusted counts did not beat this on data through 2015."""
    return pd.Series([np.busday_count(d.date(), (d + pd.offsets.MonthBegin(1)).date()) for d in index], index=index, dtype=float)


def load_features():
    """log births from START, and log regressors on a longer index for forecasting."""
    m = load_monthly()
    nw, w = newlywed(m.marr), women_30_34()
    X = pd.DataFrame({"newlywed": np.log(nw), "women": np.log(w)}).dropna()
    X["weekdays"] = np.log(weekdays(X.index))
    y = np.log(m.births[START:]).rename("y")
    assert X.index[0] <= y.index[0] and X.index[-1] >= y.index[-1]
    assert weekdays(pd.DatetimeIndex(["2015-02-01"])).iloc[0] == 20

    # no leakage: changing marriages after an origin must not move newlywed up to 12 months past it
    origin = pd.Timestamp("2020-06-01")
    bumped = m.marr.copy()
    bumped[bumped.index > origin] *= 2
    horizon_end = origin + pd.DateOffset(months=12)
    assert newlywed(bumped)[:horizon_end].dropna().equals(nw[:horizon_end].dropna()), "newlywed leaks"
    return y, X


def design(X, calendar, with_x):
    """Regressor frame for one model, or None for pure SARIMA."""
    cols = (["weekdays"] if calendar else []) + (["newlywed", "women"] if with_x else [])
    return X[cols] if cols else None


def fit(y, X, spec):
    """MLE on a rescaled problem, then results on the original scale.

    On raw scales the likelihood has a flat ridge between the women coefficient and the intercept: L-BFGS stopped early
    and rebound MAPE moved with the start (4.2~4.7% for one model). Rescaling gives the same optimum from every start.
    """
    ex = None if X is None else X.loc[y.index]
    k = None if ex is None else (1 / ex.diff(12).std()).replace(np.inf, 1.0).fillna(1.0)
    mod = SARIMAX(y * 100, exog=None if ex is None else ex * k, **spec)
    starts = [np.asarray(mod.start_params, dtype=float)]
    p, _, q = spec["order"]
    if p + q >= 3:  # ARMA(2,2)-like orders keep several optima even after rescaling
        rng = np.random.default_rng(zlib.crc32(f"{spec}{len(y)}".encode()))
        for _ in range(5):
            noise = rng.normal(0, 0.3, len(starts[0]))
            st = starts[0].copy()
            for i, n in enumerate(mod.param_names):
                if n != "sigma2":
                    st[i] = np.clip(st[i] + noise[i], -0.9, 0.9) if ".L" in n else st[i] + noise[i]
            starts.append(st)
    best, converged = None, False
    for st in starts:
        try:
            first = mod.fit(st, disp=False, maxiter=1000)
            r = mod.fit(first.params, disp=False, maxiter=1000, pgtol=1e-7, factr=1e2)
        except np.linalg.LinAlgError:  # a step left the valid region; other starts remain
            continue
        r = r if r.llf >= first.llf else first
        ok = bool(first.mle_retvals["converged"] or r.mle_retvals["converged"])
        if best is None or (ok, r.llf) > (converged, best.llf):
            best, converged = r, ok
    if best is None:
        raise np.linalg.LinAlgError(f"every start failed for {spec}")
    params = best.params.copy()
    if ex is not None:
        for c in ex.columns:
            params[c] = params[c] * k[c] / 100
    if "intercept" in params:
        params["intercept"] /= 100
    params["sigma2"] /= 1e4
    res = SARIMAX(y, exog=ex, **spec).smooth(params.values)
    res.mle_retvals = {"converged": converged}
    return res


def label(spec):
    (p, d, q), (P, D, Q, _) = spec["order"], spec["seasonal_order"]
    return f"({p},{d},{q})({P},{D},{Q})12" + ("+c" if spec.get("trend") == "c" else "")


def mape(forecast, actual):
    return float((np.abs(forecast - actual) / actual).mean() * 100)


def rmse(forecast, actual):
    return float(np.sqrt(((forecast - actual) ** 2).mean()))
