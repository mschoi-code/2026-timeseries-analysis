"""E5: province panel R1 (levels) and R3 (differences) with an age-standardized marriage rate in place of the general rate.

Wife's age-specific rates 15~54 weighted by national women of a fixed year. Report only; rules are in notes/decisions.md.
Run from sarimax_etc/: python -I scripts/province_standardized.py
"""
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))  # appended last: scripts run with python -I
sys.path.append(str(ROOT / "scripts"))
from common import REP  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from kosis_check import MARR_AGES, number, sheet, women  # noqa: E402
from province_panel import NATIONAL, SEJONG, cumulative, design, load, ols, randomization_p, report  # noqa: E402

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)
STANDARDS = (2005, 2020)  # 2005 is KOSIS's standard population for age-standardized rates; 2020 is the declared sensitivity
LAGS = (0, 1, 2, 3)
DAEJEON = "대전광역시"


def wife_rates(who="아내"):
    """{region: years x 5-year bands} of marriages per 1,000 people of that age; who="남편" gives husband rates."""
    rows = sheet("시도_성_연령별*")
    yr_head, item = rows[0], rows[1]
    out, region = {}, None
    for r in rows[2:]:
        region = r[0] or region  # the region label appears only on the first row of each block
        if r[1] in MARR_AGES:
            for j in range(2, len(r)):
                if item[j].startswith(who):
                    out.setdefault(region, {}).setdefault(int(yr_head[j]), {})[r[1]] = number(r[j])
    return {("전국" if k == "계" else k): pd.DataFrame(v).T.sort_index()[MARR_AGES] for k, v in out.items()}


def standardized(rates, year):
    w = women().loc[year, MARR_AGES]
    w = w / w.sum()
    return pd.DataFrame({k: (R * w).sum(axis=1, min_count=len(MARR_AGES)) for k, R in rates.items()})


def main():
    _, tfr, rate, _ = load()
    rates = wife_rates()
    assert NATIONAL in rates and set(tfr.columns) <= set(rates), set(tfr.columns) - set(rates)
    for k, R in rates.items():
        if k != SEJONG:
            assert R.loc[1997:2025].notna().all().all(), k  # R3 at 2001 with L3 reaches back to 1997

    old = pd.read_csv(REP / "12_panel_coefficients.csv").set_index(["spec", "term"]).b
    rows, ri = [], []
    variants = {"general rate (12)": rate, **{f"ASMR {s}": standardized(rates, s) for s in STANDARDS}}
    for vname, g in variants.items():
        for spec, start, diff, seed in (("R1 levels 2000~2023", 2000, False, 1), ("R3 differences 2001~2023", 2001, True, 3)):
            r = ols(*design(tfr, g, LAGS, start, 2023, diff=diff))
            rows += [dict(rate=vname, **x) for x in report(spec, r, LAGS)]
            b = cumulative(r)[0]
            if vname.startswith("general"):  # must reproduce reports/12 before the swap means anything
                assert abs(b - old[(spec, "L1+L2")]) < 1e-9, spec
            p, lo, hi = randomization_p(tfr, g, LAGS, start, 2023, diff, b, seed)
            ri.append(dict(rate=vname, spec=spec, l1_l2=b, ri_p=p, ri_lo=lo, ri_hi=hi))
    T = pd.DataFrame(rows).merge(pd.DataFrame(ri)[["rate", "spec", "ri_p"]].assign(term="L1+L2"), how="left", on=["rate", "spec", "term"])
    T.to_csv(REP / "17_panel_coefficients.csv", index=False)
    print(T.round(3).to_string(index=False))
    print(pd.DataFrame(ri).round(3).to_string(index=False))

    # narrative only: is Daejeon's 2022->2024 marriage-rate jump an age-structure effect?
    units = [c for c in tfr.columns if c != SEJONG]
    ch = pd.DataFrame({vname: np.log(g.loc[2024, units] / g.loc[2022, units]) * 100 for vname, g in variants.items()})
    ch["age_structure_part_2005"] = ch["general rate (12)"] - ch["ASMR 2005"]
    for band in ("25 - 29세", "30 - 34세", "35 - 39세"):
        ch[f"band {band}"] = [np.log(rates[u].loc[2024, band] / rates[u].loc[2022, band]) * 100 for u in ch.index]
    ch.to_csv(REP / "17_rate_change_22_24.csv")
    print("\nlog change 2022->2024 x100\n", ch.round(1).sort_values("general rate (12)").to_string())


if __name__ == "__main__":
    main()
