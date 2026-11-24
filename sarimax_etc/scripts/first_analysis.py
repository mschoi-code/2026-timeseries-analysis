"""E10: first analysis agreed at the team meeting: newlywed couples as the only X, the deck's split; report only.

Two designs: the deck as written (no publication lag) and real time (2 months). Rules are in notes/decisions.md.
Run from births/: python -I scripts/first_analysis.py   (BIRTHS_WORKERS sets the process count, default 6)
"""
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))  # appended last: scripts run with python -I
sys.path.append(str(ROOT / "scripts"))
import pandas as pd  # noqa: E402
import realtime as R  # noqa: E402

WORKERS = int(os.environ.get("BIRTHS_WORKERS", 6))
DESIGNS = {"deck (no lag)": 0, "real-time (2-month lag)": 2}


def models():
    out = []
    for d, spec in (("d=1", R.D1), ("d=0", R.D0)):
        for cal in (True, False):
            tag = "weekdays" if cal else "no weekdays"
            out.append(dict(name=f"{d} | SARIMA | {tag}", spec=spec, calendar=cal))
            for lo, hi in ((12, 35), (9, 32)):
                out.append(dict(name=f"{d} | newlywed {lo}~{hi} | {tag}", spec=spec, calendar=cal, kind="win", lo=lo, hi=hi))
    return out


def main():
    rows = []
    for design, lag in DESIGNS.items():
        R.PUB_LAG = lag  # read at call time; forked workers inherit it
        with ProcessPoolExecutor(WORKERS) as ex:
            res = list(ex.map(R.evaluate, models()))
        rows += [r.assign(design=design) for r, _ in res] + [R.naive_rows().assign(design=design)]
    B = pd.concat(rows, ignore_index=True)
    B.to_csv(R.REP / "22_first_analysis_by_origin.csv", index=False)
    S = B.groupby(["design", "name", "period"], sort=False)[["mape", "rmse", "mase", "bias"]].mean().reset_index()
    S.to_csv(R.REP / "22_first_analysis_summary.csv", index=False)

    rt = S[(S.design == "real-time (2-month lag)") & (S.name == "d=1 | SARIMA | weekdays")].set_index("period").mape
    old = pd.read_csv(R.REP / "09_realtime_summary.csv")
    old = old[old.name == "d=1 | no X"].set_index("period").mape
    assert abs(rt["official normal"] - old["official normal"]) < 1e-6 and abs(rt["rebound 2023-12"] - old["rebound 2023-12"]) < 1e-6, "09 not reproduced"

    pd.set_option("display.width", 220)
    for design in DESIGNS:
        part = S[(S.design == design) & S.period.isin(["official normal", "rebound 2023-12"])]
        wide = part.pivot_table(index="name", columns="period", values=["mape", "rmse"], sort=False)
        print(f"\n{design}: MAPE % and RMSE (births), normal = 8 December decisions, rebound = 20 frozen decisions\n", wide.round(2).to_string())
    print("\nreproduced 09 'd=1 | no X' in real time:", rt[["official normal", "rebound 2023-12"]].round(2).tolist())
    ref = pd.read_csv(R.REP / "08_a_rate_vs_population.csv").set_index("name").loc["d=1 | newlywed only", ["normal", "rebound"]]
    print("reference 08_a 'd=1 | newlywed only' (older code path, no lag):", ref.round(2).tolist())


if __name__ == "__main__":
    main()
