"""E1: rerun the team EDA notebook unchanged on CSVs rebuilt from the Drive files, then refit its SARIMAX with common.fit.

Report only; rules are in notes/decisions.md. Run from births/: python -I scripts/notebook_rerun.py <scratch_dir>
The notebook and its two input CSVs are copied into <scratch_dir>; nothing in the repo is changed but reports/13_*.
"""
import ast
import io
import json
import os
import shutil
import sys
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))  # appended last: scripts run with python -I
from common import EDA_71, PROPOSAL, RAW, REP, fit, load_monthly  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402  (common already set the Agg backend)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from statsmodels.stats.stattools import durbin_watson  # noqa: E402

NB = RAW / "team_drive_20261109" / "birth_forecast_eda.ipynb"
# numbers printed in the 2026-10-09 EDA PDF (text in notes/births_eda.md)
PDF = {
    ("rebound 2024-01~2026-07", "SARIMA", "MAPE"): 18.0,
    ("rebound 2024-01~2026-07", "SARIMAX + 혼인 건수", "MAPE"): 17.4,
    ("rebound 2024-01~2026-07", "SARIMAX + 신혼 부부 수 + 인구", "MAPE"): 7.1,
    ("rebound 2024-01~2026-07", "SARIMA", "2026 under (%)"): 46.0,
    ("rebound 2024-01~2026-07", "SARIMAX + 신혼 부부 수 + 인구", "2026 under (%)"): 13.0,
    ("rolling 12 ahead 2025-01~2026-07", "Naive", "MAPE"): 8.7,
    ("rolling 12 ahead 2025-01~2026-07", "수준형 SARIMA", "MAPE"): 11.1,
    ("rolling 12 ahead 2025-01~2026-07", "수준형 SARIMAX + 신혼 부부 수 + 인구", "MAPE"): 6.3,
    ("normal 2022~2023", "Holt-Winters", "MAPE"): 4.9,
    ("normal 2022~2023", "SARIMA(0,1,1)(0,1,1)12", "MAPE"): 7.2,
    ("normal 2022~2023", "Naive", "MAPE"): 9.0,
    ("normal 2022~2023", "추세+계절 더미 회귀", "MAPE"): 21.2,
    ("residual autocorrelation", "일반 회귀", "Durbin-Watson"): 0.34,
    ("residual autocorrelation", "AR(2) 오차 회귀", "Durbin-Watson"): 1.99,
    ("by horizon", "일반 회귀", "1개월 앞"): 5.1,
    ("by horizon", "AR(2) 오차 회귀", "1개월 앞"): 2.8,
    ("by horizon", "일반 회귀", "3개월 앞"): 5.4,
    ("by horizon", "AR(2) 오차 회귀", "3개월 앞"): 3.9,
    ("by horizon", "일반 회귀", "12개월 앞"): 6.7,
    ("by horizon", "AR(2) 오차 회귀", "12개월 앞"): 9.7,
    ("seasonality", "January", "deviation (%)"): 13.0,
    ("seasonality", "December", "deviation (%)"): -14.0,
    ("births per recent couple", "2026-07 vs 2020-12", "change (%)"): 18.0,
}
DECIMALS = {"Durbin-Watson": 2, "2026 under (%)": 0, "deviation (%)": 0, "change (%)": 0}  # the PDF's printed precision


def build_inputs(tmp):
    """The two CSVs the notebook reads, in its column layout, from data/raw/."""
    m = load_monthly()
    pd.DataFrame({"연도": m.index.year, "월": m.index.month, "출생아수": m.births.astype(int).values,
                  "혼인건수": m.marr.astype(int).values}).to_csv(tmp / "출생아수_혼인건수_월별.csv", index=False, encoding="utf-8-sig")
    p = pd.read_csv(RAW / "성_및_연령별_추계인구_1세별_5세별_전국.csv", header=None, dtype=str)
    years = p.iloc[2, 1:].astype(int).values
    body = p.iloc[3:].set_index(0)
    assert body.index.is_unique
    pop = pd.DataFrame({a.replace(" ", ""): pd.to_numeric(body.loc[a], errors="coerce").values for a in body.index}, index=pd.Index(years, name="연도"))
    pop.to_csv(tmp / "여성인구_연령별_연도별_추계.csv", encoding="utf-8-sig")


def run_notebook(path):
    """Execute code cells in order in one namespace, printing a cell's last expression as Jupyter would."""
    ns, log = {"__name__": "__main__"}, []
    for i, cell in enumerate(c for c in json.loads(path.read_text(encoding="utf-8"))["cells"] if c["cell_type"] == "code"):
        tree = ast.parse("".join(cell["source"]))
        last = tree.body.pop() if tree.body and isinstance(tree.body[-1], ast.Expr) else None
        buf = io.StringIO()
        with redirect_stdout(buf):
            exec(compile(tree, f"<cell {i}>", "exec"), ns)
            if last is not None:
                val = eval(compile(ast.Expression(last.value), f"<cell {i}>", "eval"), ns)
                if val is not None:
                    print(val)
        plt.close("all")
        log.append(f"--- code cell {i}\n{buf.getvalue()}")
    return ns, "\n".join(log)


def collect(ns):
    """Numbers the notebook produced, keyed like PDF."""
    mape, births, targets = ns["mape"], ns["births"], ns["targets"]
    out = {}
    for k, v in ns["res1"].items():
        out[("normal 2022~2023", k, "MAPE")] = v
    for k, row in ns["result"].iterrows():
        out[("rebound 2024-01~2026-07", k, "MAPE")] = row["MAPE(%)"]
        out[("rebound 2024-01~2026-07", k, "2026 under (%)")] = row["2026년 과소예측(%)"]
    for k, v in ns["roll"].items():
        out[("rolling 12 ahead 2025-01~2026-07", k, "MAPE")] = mape(births[targets], v)
    for L in (3, 6, 12, 18, 24):
        out[("Granger p", "혼인 → 출생", f"lag {L}")] = float(ns["fwd"][L][0]["ssr_ftest"][1])
    for L in (6, 12):
        out[("Granger p", "출생 → 혼인", f"lag {L}")] = float(ns["rev"][L][0]["ssr_ftest"][1])
    out[("residual autocorrelation", "일반 회귀", "Durbin-Watson")] = durbin_watson(ns["ols"].resid)
    out[("residual autocorrelation", "AR(2) 오차 회귀", "Durbin-Watson")] = durbin_watson(ns["ar2"].resid[12:])
    for hz, row in pd.DataFrame(ns["rows"]).T.iterrows():
        for k, v in row.items():
            out[("by horizon", k, hz)] = v
    season = ns["season"].groupby(ns["season"].index.month).mean()
    out[("seasonality", "January", "deviation (%)")] = season[1]
    out[("seasonality", "December", "deviation (%)")] = season[12]
    ratio = ns["ratio"]
    out[("births per recent couple", "2026-07 vs 2020-12", "change (%)")] = (ratio.iloc[-1] / ratio["2020-12-01"] - 1) * 100
    return out


def refit(ns):
    """Same designs as the notebook's rebound and rolling tables, estimated with common.fit."""
    y, marr, births, X_design, mape = ns["y"], ns["marr"], ns["births"], ns["X_design"], ns["mape"]
    TR, TE, act, n, targets = ns["TR"], ns["TE"], ns["act"], ns["n"], ns["targets"]
    lag_marr = np.log(marr).shift(12).rename("marr").to_frame()
    fc = {
        "SARIMA": fit(y[TR], None, PROPOSAL).get_forecast(n),
        "SARIMAX + 혼인 건수": fit(y[TR], lag_marr[TR], PROPOSAL).get_forecast(n, exog=lag_marr[TE]),
        "SARIMAX + 신혼 부부 수 + 인구": fit(y[TR], X_design[TR], EDA_71).get_forecast(n, exog=X_design[TE]),
    }
    out = {}
    for k, r in fc.items():
        f = pd.Series(np.exp(r.predicted_mean.values), index=act.index)
        out[("rebound 2024-01~2026-07", k, "MAPE")] = mape(act, f)
        out[("rebound 2024-01~2026-07", k, "2026 under (%)")] = ((act["2026"] / f["2026"] - 1) * 100).mean()
    roll = {"수준형 SARIMA": [], "수준형 SARIMAX + 신혼 부부 수 + 인구": []}
    for d in targets:
        end = d - pd.DateOffset(months=12)
        yy = y["2005-01":end]
        fut = pd.date_range(end + pd.DateOffset(months=1), d, freq="MS")
        roll["수준형 SARIMA"].append(np.exp(fit(yy, None, EDA_71).get_forecast(12).predicted_mean.iloc[-1]))
        roll["수준형 SARIMAX + 신혼 부부 수 + 인구"].append(np.exp(fit(yy, X_design["2005-01":end], EDA_71).get_forecast(12, exog=X_design.loc[fut]).predicted_mean.iloc[-1]))
    for k, v in roll.items():
        out[("rolling 12 ahead 2025-01~2026-07", k, "MAPE")] = mape(births[targets], v)
    # the 7.1% model: how far the notebook's default fit stopped from the optimum
    r_nb = ns["SARIMAX"](y[TR], exog=X_design[TR], order=(1, 0, 0), seasonal_order=(0, 1, 1, 12), trend="c").fit(disp=False)
    r_cf = fit(y[TR], X_design[TR], EDA_71)
    params = pd.DataFrame({"notebook_fit": r_nb.params, "common_fit": r_cf.params})
    params.loc["loglik"] = [r_nb.llf, r_cf.llf]
    params.loc["converged"] = [float(r_nb.mle_retvals["converged"]), float(r_cf.mle_retvals["converged"])]
    return out, params


def main():
    tmp = Path(sys.argv[1]).resolve()
    tmp.mkdir(parents=True, exist_ok=True)
    build_inputs(tmp)
    shutil.copy(NB, tmp / NB.name)
    here = os.getcwd()
    os.chdir(tmp)
    try:
        ns, log = run_notebook(tmp / NB.name)
    finally:
        os.chdir(here)
    (tmp / "notebook_output.txt").write_text(log, encoding="utf-8")
    print(log)

    got = collect(ns)
    cf, params = refit(ns)
    keys = list(dict.fromkeys(list(got) + list(PDF)))
    T = pd.DataFrame([dict(section=s, item=i, metric=m_, notebook_rerun=got.get((s, i, m_)), eda_pdf=PDF.get((s, i, m_)), common_fit=cf.get((s, i, m_))) for s, i, m_ in keys])
    T["matches_pdf"] = [None if pd.isna(p) or pd.isna(v) else round(v, DECIMALS.get(m_, 1)) == p for v, p, m_ in zip(T.notebook_rerun, T.eda_pdf, T.metric)]
    T.to_csv(REP / "13_notebook_numbers.csv", index=False)
    params.to_csv(REP / "13_notebook_71_params.csv")
    pd.set_option("display.width", 200)
    print(T.round(3).to_string(index=False))
    print(params.round(4).to_string())


if __name__ == "__main__":
    main()
