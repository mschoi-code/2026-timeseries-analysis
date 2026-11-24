"""E9: Japan's 2005->2015 fertility rebound next to Korea's 2023->2025, with the same measures; report only.

Rules are in notes/decisions.md. Run from sarimax_etc/: python -I scripts/japan_compare.py
Japanese tables: e-Stat vital statistics (final counts), CSV files kept as downloaded in data/raw/japan_estat_20261110/.
"""
import csv
import io
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))  # appended last: scripts run with python -I
sys.path.append(str(ROOT / "scripts"))
from common import RAW, REP  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from kosis_check import AGES, kitagawa, national_rates  # noqa: E402
from province_standardized import wife_rates  # noqa: E402

JP = RAW / "japan_estat_20261110"
BANDS = ["15～19歳", "20～24歳", "25～29歳", "30～34歳", "35～39歳", "40～44歳", "45～49歳"]
ORDERS = {"総　　　数": "all", "第　１　子": "1st", "第　２　子": "2nd", "第３子以上": "3rd+"}


def rows(name):
    return list(csv.reader(io.StringIO((JP / name).read_bytes().decode("cp932"))))


def num(x):
    x = x.replace("　", "").replace(" ", "").strip()
    try:
        return float(x)
    except ValueError:
        return np.nan  # "…", "-", "・" mark missing values


def year(x):
    m = re.match(r"\s*(\d{4})", x)
    return int(m.group(1)) if m else None


def by_year_rows(name, cols):
    """Tables with one row per year."""
    out = {year(r[0]): [num(v) for v in r[1:1 + len(cols)]] for r in rows(name) if len(r) > 1 and year(r[0])}  # note lines can start with a year
    return pd.DataFrame(out, index=cols).T.sort_index()


def sections(name, header_row):
    """Tables with years across and labelled blocks down: {(block, row label): Series by year}."""
    R = rows(name)
    years = [year(y) for y in R[header_row][1:]]
    out, block = {}, None
    for r in R[header_row + 1:]:
        if not r or not r[0].strip():
            continue
        vals = [num(v) for v in r[1:len(years) + 1]]
        if all(np.isnan(v) for v in vals):
            block = r[0].strip()
            continue
        out[(block, r[0].strip())] = pd.Series(vals, index=years)
    return out


# --- Japan -----------------------------------------------------------------
basic = by_year_rows("2025_4-1.csv", ["births", "male", "female", "crude", "sex_ratio", "tfr"])
s46 = sections("2025_4-6.csv", 6)
births_band = pd.DataFrame({b: s46[("出 生 数", b)] for b in BANDS})
rate_band = pd.DataFrame({b: s46[("出生率(女性人口千対)", b)] for b in BANDS})
s418 = sections("2024_4-18.csv", 8)
tfr_order = {o: pd.DataFrame({b: s418[(blk, b)] for b in BANDS}) for blk, o in ORDERS.items()}
assert np.allclose(sum(tfr_order[o] for o in ("1st", "2nd", "3rd+")).sum(axis=1), tfr_order["all"].sum(axis=1), atol=2e-3), "orders do not add up"
assert np.allclose(tfr_order["all"].sum(axis=1).round(2), basic.tfr.reindex(tfr_order["all"].index), atol=0.011), "4-18 vs 4-1 TFR"
marr_rows = rows("2025_9-1.csv")
head = next(r for r in marr_rows if r and r[0] == "" and year(r[1] or ""))
nat = next(r for r in marr_rows if r and r[0].replace("　", "") == "全国")
marriages = pd.Series([num(v) for v in nat[1:]], index=[year(y) for y in head[1:]])
first_marr = {}
block = None
for r in rows("2024_9-8.csv"):
    if r and r[0].startswith("妻"):
        block = "wife"
    elif block == "wife" and r and year(r[0]):
        first_marr[year(r[0])] = [num(v) for v in r[1:8]]
first_marr = pd.DataFrame(first_marr, index=["≤19", "20~24", "25~29", "30~34", "35~39", "40~44", "45~49"]).T
mean_age = by_year_rows("2025_9-11.csv", ["all_h", "all_w", "first_h", "first_w", "gap_all", "gap_first"])
delay = by_year_rows("2025_9-16.csv", [f"c{i}" for i in range(16)])
delay_1y = delay[["c12", "c13", "c14", "c15"]].sum(axis=1)  # 1 year or longer from moving in to registration
nonmarital = by_year_rows("2025_4-29.csv", ["total", "marital", "nonmarital", "p_total", "p_marital", "p_nonmarital"]).p_nonmarital
dur = sections("2025_4-22.csv", 7)
first_within_1y = dur[(None, "1年未満")]

YRS = [1990, 1995, 2000, 2005, 2010, 2015, 2019, 2020, 2022, 2023, 2024, 2025]
J1 = pd.DataFrame({"tfr": basic.tfr, "births": basic.births, "marriages": marriages, "first_marr_age_w": mean_age.first_w,
                   "reg_delay_1y_pct": delay_1y, "first_birth_within_1y_of_marriage_pct": first_within_1y,
                   "nonmarital_pct": nonmarital}).reindex(YRS)

# J2: births change into female-population and rate effects, women = births / rate
W = births_band / rate_band * 1000
jp = kitagawa(rate_band.loc[2005], rate_band.loc[2015], W.loc[2005], W.loc[2015])
kr = pd.read_csv(REP / "10_kitagawa_births.csv", index_col=0)
kr = kr[(kr["from"] == 2023) & (kr["to"] == 2025)][["population", "rate"]]
J2 = pd.DataFrame({"japan_2005_2015": jp.sum(), "korea_2023_2025": kr.sum()}).T
J2["actual"] = [basic.births[2015] - basic.births[2005], 254341 - 230028]

# J3: TFR change by birth order and by age band
J3_order = pd.Series({o: tfr_order[o].loc[2015].sum() - tfr_order[o].loc[2005].sum() for o in ("1st", "2nd", "3rd+")})
J3_age = tfr_order["all"].loc[2015] - tfr_order["all"].loc[2005]
kr_rates = national_rates()[AGES]
J3_age_kr = (kr_rates.loc[2025] - kr_rates.loc[2023]) * 5 / 1000
order_kr = pd.read_csv(REP / "10_birth_order.csv", index_col=0)

# J4: marriage path in the rebound
kr_marr = wife_rates()["전국"]
J4 = pd.DataFrame({"japan_first_marr_2005": first_marr.loc[2005].values, "japan_2015": first_marr.loc[2015].values,
                   "korea_all_marr_2023": kr_marr.loc[2023, ["15 - 19세"] + AGES[1:]].values,
                   "korea_2025": kr_marr.loc[2025, ["15 - 19세"] + AGES[1:]].values}, index=first_marr.columns)


# J5: cohort diagonals, Y0 = year the cohort was 15~19
def diagonal(band_values, y0, last):
    """Sum of band k read in year y0+5k; years past `last` fall back to `last`; returns (total, by 29, 30+, bands seen)."""
    vals = [band_values.loc[min(y0 + 5 * k, last)].iloc[k] if y0 + 5 * k <= last + 1 else np.nan for k in range(7)]
    v = np.array(vals)
    return np.nansum(v), np.nansum(v[:3]), np.nansum(v[3:]), int((~np.isnan(v)).sum()), np.nansum(v[:4])


J5 = []
for y0 in range(1970, 2015, 5):
    for o in ("all", "1st", "2nd", "3rd+"):
        tot, early, late, seen, by34 = diagonal(tfr_order[o], y0, 2024)
        J5.append(dict(country="japan", born=f"{y0 - 19}~{y0 - 15}", order=o, cum=tot, by_29=early, by_34=by34, from_30=late, bands_seen=seen))
kr_band = kr_rates * 5 / 1000
for y0 in range(1995, 2015, 5):
    tot, early, late, seen, by34 = diagonal(kr_band, y0, 2025)
    J5.append(dict(country="korea", born=f"{y0 - 19}~{y0 - 15}", order="all", cum=tot, by_29=early, by_34=by34, from_30=late, bands_seen=seen))
J5 = pd.DataFrame(J5)

# J6: COVID marriage deficit, the E7 counterfactuals on annual counts
fit = marriages.loc[2015:2019]
b = np.polyfit(fit.index, np.log(fit.values), 1)
after = marriages.loc[2020:2025]
cf = {"trend 2015~2019": np.exp(np.polyval(b, after.index)), "hold 2019": np.full(len(after), marriages[2019])}
kr_e7 = pd.read_csv(REP / "19_catchup_annual.csv")
J6 = []
for name, c in cf.items():
    gap = after - c
    J6.append(dict(country="japan", cf=name, deficit_20_22=gap.loc[:2022].sum(), after_23_25=gap.loc[2023:].sum(),
                   cum_2025=gap.sum(), cum_pct_of_cf=gap.sum() / c.sum() * 100))
for name, key in (("trend 2015~2019", "B count 2015~2019"), ("hold 2019", "D hold 2019")):
    g = kr_e7[(kr_e7.cf == key) & (kr_e7.year <= 2025)].set_index("year")
    J6.append(dict(country="korea", cf=name, deficit_20_22=g.gap.loc[:2022].sum(), after_23_25=g.gap.loc[2023:].sum(),
                   cum_2025=g.gap.sum(), cum_pct_of_cf=g.gap.sum() / g.counterfactual.sum() * 100))
J6 = pd.DataFrame(J6)

if __name__ == "__main__":  # outputs only; tempo_adjusted.py imports the parsed tables
    for name, frame in (("J1_series", J1), ("J2_kitagawa", J2), ("J4_marriage_rates", J4), ("J5_cohorts", J5), ("J6_covid", J6)):
        frame.to_csv(REP / f"21_{name}.csv")
    pd.DataFrame({"japan_tfr_change_2005_2015": J3_age, "korea_tfr_change_2023_2025": J3_age_kr.values}, index=BANDS).to_csv(REP / "21_J3_age.csv")
    J3_order.to_csv(REP / "21_J3_order.csv")

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 20)
    print("J1\n", J1.round(2))
    print("\nJ2 births change: population vs rate effect\n", J2.round(0))
    print("\nJ3 Japan TFR change 2005->2015 by order", J3_order.round(3).to_dict(), "total", round(J3_order.sum(), 3))
    counts = []
    for r in rows("2025_4-12.csv"):
        if r and r[0].startswith("百"):
            break  # the percentage block repeats the same years
        if r and year(r[0]):
            counts.append([year(r[0])] + [num(v) for v in r[1:7]])
    order_jp = pd.DataFrame(counts, columns=["year", "total", "1st", "2nd", "3rd", "4th", "5th+"]).set_index("year")
    order_jp["3rd+"] = order_jp[["3rd", "4th", "5th+"]].sum(axis=1)
    print("   Japan births by order 2005->2015:", (order_jp.loc[2015, ["1st", "2nd", "3rd+"]] - order_jp.loc[2005, ["1st", "2nd", "3rd+"]]).to_dict())
    print("   Korea births by order 2023->2025:", (order_kr.loc[2025, ["first", "second", "third_plus"]] - order_kr.loc[2023, ["first", "second", "third_plus"]]).to_dict())
    print("   TFR change by age band\n", pd.DataFrame({"japan_05_15": J3_age.values, "korea_23_25": J3_age_kr.values}, index=BANDS).round(3))
    print("\nJ4 marriage rates per 1,000 women (Japan first marriages, Korea all marriages)\n", J4.round(1))
    print("\nJ5 cohort cumulative fertility (5-year diagonals)\n", J5.round(3).to_string(index=False))
    print("\nJ6 COVID marriages\n", J6.round(1).to_string(index=False))
