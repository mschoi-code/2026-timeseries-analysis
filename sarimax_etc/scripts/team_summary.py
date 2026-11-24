"""E6: tables for the team summary, read from earlier results, plus a rerun check of two real-time models.

Report only; rules are in notes/decisions.md (E6). Writes reports/18_bridge.csv, 18_realtime_main.csv, 18_scenarios_2031.csv.
"""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))  # appended last: scripts run with python -I
import realtime as rt  # noqa: E402  (module level only loads data)
import pandas as pd  # noqa: E402

REP = rt.REP
WINDOWS = ("12~35", "9~32")
MAIN = ["seasonal naive"] + [f"{d} | {w}" for d in ("d=1", "d=0") for w in WINDOWS]
EDA_RT = "cand | proposal | (1,0,0)(0,1,1)12+c"  # the EDA notebook's spec in the real-time design (E2c)


def rerun(name):
    d, w = name.split(" | ")
    lo, hi = map(int, w.split("~"))
    R, _ = rt.evaluate(dict(name=name, spec=rt.D1 if d == "d=1" else rt.D0, women="safe", kind="win", lo=lo, hi=hi))
    return name, R.groupby("period", sort=False).mape.mean()


if __name__ == "__main__":
    S = pd.read_csv(REP / "09_realtime_summary.csv")
    mape = S.set_index(["name", "period"]).mape

    # the pipeline still gives the 11-06 numbers
    with ProcessPoolExecutor(2) as ex:
        for name, got in ex.map(rerun, ["d=1 | 12~35", "d=1 | 9~32"]):
            diff = (got - mape[name].reindex(got.index)).abs().max()
            print(f"rerun {name}: max |MAPE diff| vs 09 = {diff:.2e}")
            assert diff < 1e-6, name

    leak = pd.read_csv(REP / "14_notebook_leak.csv").set_index(["variant", "fit"]).mape
    a, b = "(a) notebook: actual marriages, known to 2023-12", "(b) marriages after 2023-12 = same month a year earlier"
    bridge = pd.DataFrame([
        dict(step=1, design="2023-12 origin, 31 months ahead, actual future marriages", fit="notebook", spec="EDA", rebound_mape=leak[a, "notebook fit"]),
        dict(step=2, design="same", fit="converged (common.fit)", spec="EDA", rebound_mape=leak[a, "common.fit"]),
        dict(step=3, design="future marriages = same month a year earlier", fit="notebook", spec="EDA", rebound_mape=leak[b, "notebook fit"]),
        dict(step=3, design="future marriages = same month a year earlier", fit="converged (common.fit)", spec="EDA", rebound_mape=leak[b, "common.fit"]),
        dict(step=4, design="real-time: 20 monthly decisions, 12 months ahead, data to o-2", fit="converged", spec="EDA", rebound_mape=mape[EDA_RT, "rebound 2023-12"], normal_mape=mape[EDA_RT, "official normal"]),
    ] + [dict(step=5, design="real-time, same", fit="converged", spec=n, rebound_mape=mape[n, "rebound 2023-12"], normal_mape=mape[n, "official normal"]) for n in MAIN[1:]])
    bridge.to_csv(REP / "18_bridge.csv", index=False)
    print(bridge.round(2).to_string(index=False))

    blocks = [p for p in S.period.unique() if p.endswith("start")]
    rows = []
    for n in MAIN:
        g = S[S.name == n].set_index("period")
        rows.append(dict(name=n, normal_mape=g.mape["official normal"], normal_mase=g.mase["official normal"],
                         rebound_mape=g.mape["rebound 2023-12"], rebound_mase=g.mase["rebound 2023-12"], rebound_bias=g.bias["rebound 2023-12"],
                         blocks_mape_min=g.mape[blocks].min(), blocks_mape_max=g.mape[blocks].max()))
    T = pd.DataFrame(rows)
    T.to_csv(REP / "18_realtime_main.csv", index=False)
    print(T.round(2).to_string(index=False))

    # 2031: model scenarios, with the structural part replaced by the fixed-rate path (D4)
    fr = pd.read_csv(REP / "10_fixed_rate_path.csv", index_col=0).births_at_2025_rates
    ratio = fr[2031] / fr[2026]
    sc = pd.read_csv(REP / "09_scenarios.csv")
    y31 = sc[sc.year == 2031].pivot_table(index=["order", "window"], columns=["rate", "population"], values="births", sort=False)
    actual25 = float(rt.level["2025"].sum())
    out = []
    for (d, w), r in y31.iterrows():
        for rate in ("hold", "decline"):
            adj = r[(rate, "frozen")] * ratio
            out.append(dict(order=d, window=w, rate=rate, model_projected=r[(rate, "projected")], frozen_2026_07=r[(rate, "frozen")],
                            fixed_rate_structure=adj, vs_2025_pct=(adj / actual25 - 1) * 100))
    O = pd.DataFrame(out)
    O.to_csv(REP / "18_scenarios_2031.csv", index=False)
    print(f"actual 2025 births {actual25:,.0f}; fixed-rate structure 2026->2031 {(ratio - 1) * 100:+.2f}%")
    print(O.round(1).to_string(index=False))
