"""E17: HistGradientBoosting benchmark on E10's origins and designs, as a comparison row only; report only.

Direct multi-horizon: one pooled model predicts log births_t - log births_b (b = latest known same month) with h as a feature.
Rules are in notes/decisions.md. Run from sarimax_etc/: python -I scripts/ml_benchmark.py   (BIRTHS_WORKERS, default 6)
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")  # before sklearn loads OpenMP: each worker stays on one core
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))  # appended last: scripts run with python -I
sys.path.append(str(ROOT / "scripts"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402
import realtime as R  # noqa: E402

WORKERS = int(os.environ.get("BIRTHS_WORKERS", 6))
DESIGNS = {"deck (no lag)": 0, "real-time (2-month lag)": 2}
VARIANTS = {"HGB | lags": ["h", "month", "gap", "g1", "g3", "g12"], "HGB | lags + newlywed 12~35 + weekdays": ["h", "month", "gap", "g1", "g3", "g12", "nw", "wd"]}
mo = R.mo
y = R.y  # log births, team file, 2005-01~2026-07


def table(lag):
    """One row per (anchor a = last known month, h); every feature uses only data known at a."""
    yoy = y - y.shift(12)
    last = R.BLOCKS["rebound 2023-12"]
    rows = []
    for a in pd.date_range("2007-01-01", pd.Timestamp(last) + mo(19 - lag), freq="MS"):
        o = a + mo(lag)
        nw = R.window(12, 35, R.extend_marr(a, o + mo(12)))
        g = dict(g1=yoy[a], g3=yoy[a - mo(2):a].mean(), g12=yoy[a - mo(11):a].mean())
        for h in range(1, 13):
            t = o + mo(h)
            b = t - mo(12) if t - mo(12) <= a else t - mo(24)
            rows.append(dict(a=a, t=t, b=b, h=h, month=t.month, gap=(t.year - b.year) * 12 + t.month - b.month, **g,
                             nw=nw[t] - nw[b], wd=R.log_wd[t] - R.log_wd[b], target=y.get(t, np.nan) - y[b]))
    T = pd.DataFrame(rows)
    assert T.drop(columns="target").notna().all().all()
    return T


def model_at(T, cols, known_to):
    tr = T[T.t <= known_to]
    assert tr.target.notna().all() and len(tr) > 500
    return HistGradientBoostingRegressor(random_state=0).fit(tr[cols], tr.target)


def forecast(T, mdl, cols, o, lag):
    p = T[T.a == o - mo(lag)].sort_values("h")
    assert len(p) == 12
    return pd.Series(np.exp(y[p.b].values + mdl.predict(p[cols])), index=pd.DatetimeIndex(p.t.values))


def evaluate(job):
    design, lag, name = job
    cols, T, rows = VARIANTS[name], table(lag), []
    for o in R.NORMAL:
        f = forecast(T, model_at(T, cols, o - mo(lag)), cols, o, lag)
        rows.append(dict(period="official normal", o=o, **R.metrics(f, R.mase_scale(o - mo(lag)))))
    for label, start in R.BLOCKS.items():
        o0 = pd.Timestamp(start)
        mdl, scale = model_at(T, cols, o0 - mo(lag)), R.mase_scale(o0 - mo(lag))  # frozen like the SARIMA blocks
        for o in pd.date_range(o0, periods=20, freq="MS"):
            rows.append(dict(period=label, o=o, **R.metrics(forecast(T, mdl, cols, o, lag), scale)))
    return pd.DataFrame(rows).assign(name=name, design=design)


def main():
    jobs = [(d, lag, n) for d, lag in DESIGNS.items() for n in VARIANTS]
    with ProcessPoolExecutor(min(WORKERS, len(jobs))) as ex:
        B = pd.concat(list(ex.map(evaluate, jobs)), ignore_index=True)
    B.to_csv(R.REP / "29_ml_by_origin.csv", index=False)
    S = B.groupby(["design", "name", "period"], sort=False)[["mape", "rmse", "mase", "bias"]].mean().reset_index()
    S.to_csv(R.REP / "29_ml_summary.csv", index=False)

    E10 = pd.read_csv(R.REP / "22_first_analysis_summary.csv")
    keep = ["seasonal naive", "d=1 | SARIMA | no weekdays", "d=1 | newlywed 12~35 | no weekdays", "d=1 | newlywed 9~32 | no weekdays",
            "d=0 | newlywed 12~35 | no weekdays", "d=1 | newlywed 12~35 | weekdays"]
    both = pd.concat([E10[E10.name.isin(keep)], S], ignore_index=True)
    pd.set_option("display.width", 220)
    for design in DESIGNS:
        part = both[(both.design == design) & both.period.isin(["official normal", "rebound 2023-12"])]
        print(f"\n{design}: MAPE % and bias %, E10 rows for reference\n",
              part.pivot_table(index="name", columns="period", values=["mape", "bias"], sort=False).round(2).to_string())


if __name__ == "__main__":
    main()
