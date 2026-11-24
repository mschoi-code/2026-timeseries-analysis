"""E8: did the 2023~2025 marriage surplus come from the cohorts that lost marriages in 2020~2022? Report only.

Rules are in notes/decisions.md. Run from births/: python -I scripts/marriage_cohorts.py
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))  # appended last: scripts run with python -I
sys.path.append(str(ROOT / "scripts"))
from common import REP  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from kosis_check import NEW, annual_item, number, sheet, women  # noqa: E402

AGES = range(22, 40)
FIT = range(2015, 2020)
AFTER = range(2020, 2026)
GROUPS = {"1981~85 (35~39 in 2020)": range(1981, 1986), "1986~89 (31~34)": range(1986, 1990),
          "1990~93 (27~30)": range(1990, 1994), "1994~96 (24~26)": range(1994, 1997), "1997~2000 (20~23)": range(1997, 2001)}

raw = (NEW / "101_DT_1B83A02_20261107182853.xls").read_bytes().decode("euc-kr", "replace")  # same reader as kosis_check.main
grid = [re.findall(r"<Data[^>]*>(.*?)</Data>", row, flags=re.S) for row in re.findall(r"<Row[^>]*>(.*?)</Row>", raw, flags=re.S)]
head = next(r for r in grid if r[:2] == ["남편의 연령별", "아내의 연령별"])
YEARS = [int(y.split()[0]) for y in head[4:]]
table = {(r[0], r[1]): pd.Series([number(x) for x in r[4:]], index=YEARS) for r in grid if len(r) == len(head) and r[2].startswith("혼인")}
wife = pd.DataFrame({k[1]: v for k, v in table.items() if k[0] == "계"})
husband = pd.DataFrame({k[0]: v for k, v in table.items() if k[1] == "계"})
assert (wife.drop(columns="계").sum(axis=1) == wife["계"]).all(), "wife ages do not add up"

W5 = women()
births = annual_item("출생아수__*", "출생아수(명)")


def women_at(a, t):
    """Women aged a in year t: the 5-year band split by the relative size of its five birth cohorts."""
    lo = a - a % 5
    return W5.loc[t, f"{lo} - {lo + 4}세"] * births[t - a] / births[[t - x for x in range(lo, lo + 5)]].sum()


def age_value(label):
    if label in ("계", "미상"):
        return np.nan
    nums = [int(n) for n in re.findall(r"\d+", label)]
    return nums[0] - 1 if "미만" in label else nums[0] + 2 if "이상" in label else sum(nums) / len(nums)


def mean_age(frame):
    cols = [c for c in frame.columns if not np.isnan(age_value(c))]
    return frame[cols].mul([age_value(c) for c in cols]).sum(axis=1) / frame[cols].sum(axis=1)


# check: own band rates against the official wife rates (KOSIS, per 1,000 women)
rows = sheet("시도_성_연령별*")
official = {}
region = None
for r in rows[2:]:
    region = r[0] or region
    if region == "계" and r[1] in ("25 - 29세", "30 - 34세", "35 - 39세"):
        official[r[1]] = {int(rows[0][j]): number(r[j]) for j in range(2, len(r)) if rows[1][j].startswith("아내")}
check = pd.DataFrame({band: {t: wife[[f"{a}세" for a in range(int(band[:2]), int(band[:2]) + 5)]].loc[t].sum() / W5.loc[t, band] * 1000 / official[band][t] - 1
                             for t in range(2010, 2026)} for band in official}) * 100

M = pd.DataFrame({a: wife[f"{a}세"] for a in AGES}).loc[2010:2025]
W = pd.DataFrame({a: [women_at(a, t) for t in M.index] for a in AGES}, index=M.index)
rate = M / W * 1000
t = np.array(FIT)
slope, icpt = np.polyfit(t, np.log(rate.loc[FIT]).values, 1)
trend = pd.DataFrame(np.exp(icpt + np.outer(np.array(AFTER), slope)), index=AFTER, columns=AGES)
hold = pd.DataFrame([rate.loc[2019].values] * len(AFTER), index=AFTER, columns=AGES)

out, annual = [], {}
for cf_name, cf in (("trend 2015~2019", trend), ("hold 2019", hold)):
    gap = M.loc[AFTER] - cf * W.loc[AFTER] / 1000
    annual[cf_name] = gap.sum(axis=1)
    long = gap.stack().rename("gap").reset_index().rename(columns={"level_0": "year", "level_1": "age"})
    long["cohort"] = long.year - long.age
    for g, cohorts in GROUPS.items():
        sel = long[long.cohort.isin(cohorts)]
        size = sum(women_at(2022 - c, 2022) for c in cohorts)
        d, s = sel[sel.year <= 2022].gap.sum(), sel[sel.year >= 2023].gap.sum()
        out.append(dict(cf=cf_name, cohorts=g, deficit_20_22=d, surplus_23_25=s, net=d + s, women_2022=size,
                        deficit_per_1000=d / size * 1000, surplus_per_1000=s / size * 1000, net_per_1000=(d + s) / size * 1000,
                        ages_seen=f"{2020 - max(cohorts)}~{min(39, 2025 - min(cohorts))}"))

ages = pd.DataFrame({"wife": mean_age(wife), "husband": mean_age(husband)}).loc[2010:2025]
# added after the first results: ages 22~39 only (fewer remarriages), and the rate schedule's mean (no population structure)
ages["wife_22_39"] = (M * np.array(AGES)).sum(axis=1) / M.sum(axis=1)
ages["wife_22_39_rate"] = (rate * np.array(AGES)).sum(axis=1) / rate.sum(axis=1)
fitted = {k: np.polyval(np.polyfit(t, ages.loc[FIT, k], 1), ages.index) for k in list(ages)}
ages = ages.assign(**{f"{k}_vs_trend": ages[k] - fitted[k] for k in list(fitted)})

C = pd.DataFrame(out)
C.to_csv(REP / "20_cohort_gaps.csv", index=False)
ages.to_csv(REP / "20_mean_age.csv")
check.to_csv(REP / "20_rate_check.csv")

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 20)
print("own band rate / official - 1 (%)\n", check.round(1).loc[[2010, 2015, 2019, 2020, 2022, 2023, 2024, 2025]])
assert check.abs().max().max() < 5, "denominator far from the official rates"
print("\nmarriage rate per 1,000 women, selected ages\n", rate[[24, 27, 30, 33, 36, 39]].loc[2015:].round(1))
print("\nannual gap, ages 22~39 (E7 B, all ages: 2020 -13418 ... 2025 +68688)\n", pd.DataFrame(annual).round(0))
print("\ncohort gaps\n", C.drop(columns="women_2022").round(1).to_string(index=False))
print("\nmean age at marriage\n", ages.round(2).loc[2015:])
