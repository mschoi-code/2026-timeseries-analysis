"""E3: the notebook's trend + seasonal dummy regression and Holt-Winters, plus a multiplicative Holt-Winters, in the
real-time design of reports/09 (known to o-2, 8 December decisions, blocks of 20 frozen decisions). Report only.

These are notebook versions; teammates' final settings may differ. Rules are in notes/decisions.md.
Run from births/ after notebook_leak.py: python -I scripts/team_models_realtime.py
"""
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))  # appended last: scripts run with python -I
sys.path.append(str(ROOT / "scripts"))
from common import REP  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.api as sm  # noqa: E402
from statsmodels.tsa.holtwinters import ExponentialSmoothing  # noqa: E402
from realtime import BLOCKS, NORMAL, PUB_LAG, level, mase_scale, metrics, mo, y  # noqa: E402

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)
MODELS = ("trend + seasonal dummies (log)", "Holt-Winters additive (log)", "Holt-Winters multiplicative (levels)")


def dummies(idx):
    """The notebook's X_reg: constant, months since 1981-01, 11 month dummies (January dropped)."""
    t = (idx.year - 1981) * 12 + idx.month - 1
    return np.column_stack([np.ones(len(idx)), t, (idx.month.values[:, None] == np.arange(2, 13)).astype(float)])


def holt_winters(series, seasonal, frozen=None):
    """Fit as the notebook does, or run the filter with every parameter of an earlier fit (initial states included)."""
    if frozen is None:
        return ExponentialSmoothing(series, trend="add", seasonal=seasonal, seasonal_periods=12).fit()
    p = frozen
    mod = ExponentialSmoothing(series, trend="add", seasonal=seasonal, seasonal_periods=12, initialization_method="known",
                               initial_level=p["initial_level"], initial_trend=p["initial_trend"], initial_seasonal=p["initial_seasons"])
    return mod.fit(smoothing_level=p["smoothing_level"], smoothing_trend=p["smoothing_trend"], smoothing_seasonal=p["smoothing_seasonal"], optimized=False)


def path(name, o, frozen=None):
    """Parameters and the 12 scored months o+1..o+12 from data known to o-2; frozen reuses earlier parameters."""
    known_to = o - mo(PUB_LAG)
    tg = pd.date_range(o + mo(1), periods=12, freq="MS")
    fut = pd.date_range(known_to + mo(1), tg[-1], freq="MS")
    if name.startswith("trend"):
        tr = y[:known_to]
        params = sm.OLS(tr.values, dummies(tr.index)).fit().params if frozen is None else frozen
        f = pd.Series(np.exp(dummies(fut) @ params), index=fut)
    else:
        log = "log" in name
        r = holt_winters(y[:known_to] if log else level[:known_to], "add" if log else "mul", frozen)
        params = r.params if frozen is None else frozen
        f = r.forecast(len(fut))
        f = pd.Series(np.exp(f.values) if log else f.values, index=fut)
    return params, f[tg]


def evaluate(name):
    rows = []
    for o in NORMAL:
        _, f = path(name, o)
        rows.append(dict(period="official normal", o=o, **metrics(f, mase_scale(o - mo(PUB_LAG)))))
    for label, start in BLOCKS.items():
        o0 = pd.Timestamp(start)
        scale = mase_scale(o0 - mo(PUB_LAG))
        p0, f0 = path(name, o0)
        _, again = path(name, o0, p0)
        assert np.allclose(again.values, f0.values, rtol=1e-6), (name, label)  # frozen run reproduces the fit it froze
        for o in pd.date_range(o0, periods=20, freq="MS"):
            _, f = path(name, o, p0)
            rows.append(dict(period=label, o=o, **metrics(f, scale)))
    return pd.DataFrame(rows).assign(name=name)


def main():
    with ProcessPoolExecutor(len(MODELS)) as ex:
        R = pd.concat(list(ex.map(evaluate, MODELS)), ignore_index=True)
    R.to_csv(REP / "15_by_origin.csv", index=False)
    cols = ["mape", "mase", "rmse", "bias", "ape12"]
    S = R.groupby(["name", "period"], sort=False)[cols].mean().reset_index()

    old = pd.read_csv(REP / "09_realtime_summary.csv")
    ref = {"seasonal naive": "seasonal naive (= notebook Naive)", "d=1 | 12~35": "main: d=1 (0,1,1)(0,1,1)12 + weekdays + newlywed 12~35 + women",
           "d=0 | 12~35": "d=0 alternative: (1,0,0)(0,1,1)12+c + weekdays + newlywed 12~35 + women"}
    base = old[old.name.isin(ref)].assign(name=lambda d: d.name.map(ref))
    e2c = pd.read_csv(REP / "14_realtime_by_origin.csv").groupby("period", sort=False)[cols].mean().reset_index().assign(name="notebook SARIMAX, real-time (E2c): d=0, no weekdays")
    T = pd.concat([base, e2c, S], ignore_index=True)[["name", "period"] + cols]
    T.to_csv(REP / "15_summary.csv", index=False)
    order = ["official normal"] + list(BLOCKS)
    for metric in ("mape", "mase"):
        print(f"\n{metric.upper()} by period (real-time design)\n", T.pivot_table(index="name", columns="period", values=metric, sort=False)[order].round(2).to_string())


if __name__ == "__main__":
    main()
