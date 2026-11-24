"""E2: how much of the notebook's 7.1% rebound MAPE comes from future marriages and the publication lag. Report only.

Same model throughout: (1,0,0)(0,1,1)12 + c, log newlywed 12~35 + log women 30~34, no weekdays.
Rules are in notes/decisions.md. Run from sarimax_etc/ after notebook_rerun.py: python -I scripts/notebook_leak.py
"""
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))  # appended last: scripts run with python -I
sys.path.append(str(ROOT / "scripts"))
from common import EDA_71, REP, fit, women_30_34  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from statsmodels.tsa.statespace.sarimax import SARIMAX  # noqa: E402
from realtime import D0, PUB_LAG, evaluate, extend_marr, m, mo  # noqa: E402

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)
ORIGIN = pd.Timestamp("2023-12-01")
TE = pd.date_range("2024-01-01", "2026-07-01", freq="MS")
TARGETS = pd.date_range("2025-01-01", "2026-07-01", freq="MS")  # the notebook's rolling table
births = m.births
y = np.log(births)
log_w = np.log(women_30_34()).reindex(m.index)
assert D0 == EDA_71


def design(marr):
    """The notebook's X_design: log of marriages 12~35 months earlier, log women 30~34."""
    nw = marr.reindex(m.index).shift(12).rolling(24).sum()
    return pd.concat([np.log(nw), log_w], axis=1, keys=["newlywed", "f30_34"])


def forecast(known_to, X, steps, how):
    tr = y["2005-01":known_to]
    fut = pd.date_range(known_to + mo(1), periods=steps, freq="MS")
    if how == "notebook fit":
        r = SARIMAX(tr, exog=X.loc[tr.index], order=(1, 0, 0), seasonal_order=(0, 1, 1, 12), trend="c").fit(disp=False)
    else:
        r = fit(tr, X.loc[tr.index], EDA_71)
    return pd.Series(np.exp(r.get_forecast(steps, exog=X.loc[fut]).predicted_mean.values), index=fut)


def scores(f, a):
    ape = (f - a).abs() / a * 100
    h = np.arange(1, len(a) + 1)
    return dict(mape=ape.mean(), mape_h1_12=ape[h <= 12].mean(), mape_h13_24=ape[(h > 12) & (h <= 24)].mean(), mape_h25_31=ape[h > 24].mean(),
                bias=((f / a - 1) * 100).mean(), under_2026=((a["2026"] / f["2026"] - 1) * 100).mean())


def main():
    rows = []
    known_lag = ORIGIN - mo(PUB_LAG)
    variants = {
        "(a) notebook: actual marriages, known to 2023-12": (ORIGIN, design(m.marr)),
        "(b) marriages after 2023-12 = same month a year earlier": (ORIGIN, design(extend_marr(ORIGIN, TE[-1]))),
        "(b+lag) as (b), data known to 2023-10": (known_lag, design(extend_marr(known_lag, TE[-1]))),
    }
    # self-check: (a) and (b) share every input up to 12 months past the origin
    Xa, Xb = variants["(a) notebook: actual marriages, known to 2023-12"][1], variants["(b) marriages after 2023-12 = same month a year earlier"][1]
    assert Xa[:ORIGIN + mo(12)].equals(Xb[:ORIGIN + mo(12)]) and not Xa.loc[TE].equals(Xb.loc[TE])
    a = births[TE]
    for name, (known_to, X) in variants.items():
        for how in ("notebook fit", "common.fit"):
            f = forecast(known_to, X, len(pd.date_range(known_to + mo(1), TE[-1], freq="MS")), how)[TE]
            rows.append(dict(design="fixed origin 2023-12, 31 months", variant=name, fit=how, **scores(f, a)))

    # the notebook's rolling 12-month table, then the same with a two-month publication lag
    for how in ("notebook fit", "common.fit"):
        for name, lag in (("rolling 12 ahead, no lag (notebook)", 0), ("rolling, data known to target-14 (lag 2), filled marriages", PUB_LAG)):
            pred = []
            for d in TARGETS:
                known_to = d - mo(12 + lag)
                pred.append(forecast(known_to, design(extend_marr(known_to, d)), 12 + lag, how).iloc[-1])
            f = pd.Series(pred, index=TARGETS)
            rows.append(dict(design="rolling 2025-01~2026-07", variant=name, fit=how, mape=((f - births[TARGETS]).abs() / births[TARGETS]).mean() * 100,
                             bias=((f / births[TARGETS] - 1) * 100).mean()))
    for name, back in (("seasonal naive, no lag (notebook)", 12), ("seasonal naive, lag 2", 24)):
        f = pd.Series([births[d - mo(back)] for d in TARGETS], index=TARGETS)
        rows.append(dict(design="rolling 2025-01~2026-07", variant=name, fit="-", mape=((f - births[TARGETS]).abs() / births[TARGETS]).mean() * 100,
                         bias=((f / births[TARGETS] - 1) * 100).mean()))

    # (c) the real-time design of reports/09: known to o-2, 20 frozen decisions per block, 12 months scored
    R, info = evaluate(dict(name="(c) real-time d=0 12~35 no weekdays", spec=D0, women="safe", kind="win", lo=12, hi=35, calendar=False))
    R.to_csv(REP / "14_realtime_by_origin.csv", index=False)
    S = R.groupby("period", sort=False)[["mape", "mase", "rmse", "bias", "ape12"]].mean()
    old = pd.read_csv(REP / "09_realtime_summary.csv").set_index(["name", "period"])
    for period, v in S.mape.items():  # the same model already ran in realtime.py as a candidate
        assert abs(v - old.mape[("cand | proposal | (1,0,0)(0,1,1)12+c", period)]) < 0.01, period
    for period, r in S.iterrows():
        rows.append(dict(design="real-time (09)", variant=f"(c) {period}", fit="common.fit", **r.to_dict()))
    T = pd.DataFrame(rows)

    e1 = pd.read_csv(REP / "13_notebook_numbers.csv").set_index(["section", "item", "metric"])
    key = ("rebound 2024-01~2026-07", "SARIMAX + 신혼 부부 수 + 인구", "MAPE")
    a_nb = T[(T.variant.str.startswith("(a)")) & (T.fit == "notebook fit")].mape.iloc[0]
    a_cf = T[(T.variant.str.startswith("(a)")) & (T.fit == "common.fit")].mape.iloc[0]
    assert round(a_nb, 1) == e1.notebook_rerun[key] and abs(a_cf - e1.common_fit[key]) < 1e-6, "E2 (a) must reproduce E1"
    T.to_csv(REP / "14_notebook_leak.csv", index=False)
    print(T.round(2).to_string(index=False))
    print("coefficients, (c) rebound block:", {k: round(v, 3) for k, v in info.items() if k != "name"})


if __name__ == "__main__":
    main()
