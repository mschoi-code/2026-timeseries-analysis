"""E13: 2031 scenarios and 5-year accuracy for the newlywed-only models of the first analysis; report only.

Rules are in notes/decisions.md. Run from births/: python -I scripts/scenarios_newlywed.py   (BIRTHS_WORKERS, default 6)
"""
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))  # appended last: scripts run with python -I
sys.path.append(str(ROOT / "scripts"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import realtime as R  # noqa: E402

WORKERS = int(os.environ.get("BIRTHS_WORKERS", 6))
WINDOWS = ((12, 35), (9, 32))
ORIGINS = [pd.Timestamp(f"{yr}-12-01") for yr in range(2015, 2021)]
ACTUAL = R.level.groupby(R.level.index.year).sum()


def backtest(model):
    rows = []
    for o in ORIGINS:
        r, f = R.path(None, model, o, months=60)
        assert r.mle_retvals["converged"], (model["name"], o)
        for yr, v in f.groupby(f.index.year).sum().items():
            rows.append(dict(name=model["name"], origin=o.year, ahead=yr - o.year, error=(v / ACTUAL[yr] - 1) * 100))
    return rows


def scen(args):
    lo, hi, women, calendar = args
    return R.scenarios(lo, hi, women=women, calendar=calendar)[0].assign(model="current" if women else "newlywed only")


def main():
    models = [dict(name=f"{d} | newlywed {lo}~{hi}", spec=spec, calendar=False, kind="win", lo=lo, hi=hi) for d, spec in (("d=1", R.D1), ("d=0", R.D0)) for lo, hi in WINDOWS]
    check = dict(name="check: d=1 | 12~35 | current", spec=R.D1, women="safe", kind="win", lo=12, hi=35)
    jobs = [(lo, hi, False, False) for lo, hi in WINDOWS] + [(12, 35, True, True)]
    with ProcessPoolExecutor(WORKERS) as ex:
        bt = [r for part in ex.map(backtest, models + [check]) for r in part]
        sc = list(ex.map(scen, jobs))
    B = pd.DataFrame(bt)
    S = pd.concat(sc, ignore_index=True)

    old = pd.read_csv(R.REP / "16_backtest_annual.csv")
    old = old[(old.name == "d=1 | 12~35 | current") & old.origin.isin([o.year for o in ORIGINS])].set_index(["origin", "ahead"]).error
    new = B[B.name == check["name"]].set_index(["origin", "ahead"]).error
    # same inputs, columns in another order: the optimizer lands on the same optimum (llf equal to 1e-7), forecasts differ ~3e-5%
    assert (new - old.loc[new.index]).abs().max() < 1e-3, "16 not reproduced"
    old9 = pd.read_csv(R.REP / "09_scenarios.csv")
    chk = S[S.model == "current"].merge(old9[old9.window == "12~35"], on=["order", "window", "rate", "population", "year"], suffixes=("", "_old"))
    assert len(chk) and (chk.births - chk.births_old).abs().max() < 1e-3, "09 scenarios not reproduced"

    fr = pd.read_csv(R.REP / "10_fixed_rate_path.csv", index_col=0).births_at_2025_rates
    ratio = fr[2031] / fr[2026]
    actual25 = ACTUAL[2025]
    y31 = S[(S.year == 2031) & (S.model == "newlywed only")].pivot_table(index=["order", "window"], columns=["rate", "population"], values="births", sort=False)
    out = []
    for (d, w), r in y31.iterrows():
        for rate in ("hold", "decline"):
            adj = r[(rate, "frozen")] * ratio
            out.append(dict(order=d, window=w, rate=rate, model_projected=r[(rate, "projected")], fixed_rate_structure=adj, vs_2025_pct=(adj / actual25 - 1) * 100))
    O = pd.DataFrame(out)
    B = B[B.name != check["name"]]
    B.to_csv(R.REP / "25_backtest_newlywed.csv", index=False)
    S.to_csv(R.REP / "25_scenarios_paths.csv", index=False)
    O.to_csv(R.REP / "25_scenarios_2031.csv", index=False)

    pd.set_option("display.width", 200)
    print("reproduced 16 (d=1 | 12~35 | current, 30 cells) and 09 scenarios (12~35, current)")
    print(f"\nfixed-rate structure 2026->2031 {(ratio - 1) * 100:+.2f}%, actual 2025 births {actual25:,.0f}")
    print("\n2031 births, newlywed only, no weekdays (thousands; % vs 2025)\n",
          O.assign(k=O.fixed_rate_structure / 1000, mp=O.model_projected / 1000).pivot_table(index=["order", "window"], columns="rate", values=["k", "vs_2025_pct", "mp"], sort=False).round(1).to_string())
    print("\nannual births, mean absolute error %, December 2015~2020 origins\n",
          B.assign(a=B.error.abs()).pivot_table(index="name", columns="ahead", values="a", sort=False).round(1).to_string())
    print("\nmean signed error %\n", B.pivot_table(index="name", columns="ahead", values="error", sort=False).round(1).to_string())


if __name__ == "__main__":
    main()
