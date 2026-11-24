"""E12: Bongaarts-Feeney tempo-adjusted TFR for Korea and Japan, by birth order; report only.

TFR'_i(t) = TFR_i(t) / (1 - r_i(t)), r_i(t) = centred yearly change of the mean age at birth of order i.
Rules are in notes/decisions.md. Run from sarimax_etc/: python -I scripts/tempo_adjusted.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))  # appended last: scripts run with python -I
sys.path.append(str(ROOT / "scripts"))
from common import REP  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import japan_compare as JC  # noqa: E402
from kosis_check import AGES, national_rates, number, sheet, women  # noqa: E402

MIDS = np.arange(17.5, 50, 5)  # 15~19 ... 45~49
ORD = ("1st", "2nd", "3rd+")


def r_of(mean_age):
    """Centred yearly change; the last year uses the backward change. Needs consecutive years."""
    r = (mean_age.shift(-1) - mean_age.shift(1)) / 2
    r.iloc[-1] = mean_age.iloc[-1] - mean_age.iloc[-2]
    return r


def adjust(tfr_by_order, mean_by_order):
    r = pd.DataFrame({o: r_of(mean_by_order[o]) for o in ORD})
    return (tfr_by_order[list(ORD)] / (1 - r)).sum(axis=1, min_count=len(ORD)), r  # NaN, not 0, where r is missing


# --- Korea: births by order x 5-year band, women by band, official TFR --------------------------
rows = sheet("성_모의*")
years = [int(y) for y in rows[0][3:]]
births = {}
sex = age = None
for r in rows[1:]:
    r = r + [None] * (3 - len(r))
    sex, age = r[0] or sex, r[1] or age
    if sex != "계" or r[2] in (None, "미상", "총계"):
        continue
    band = {"15세미만": "15 - 19세", "50세이상": "45 - 49세"}.get(age, age)
    if band not in AGES:
        continue
    o = {"1아": "1st", "2아": "2nd"}.get(r[2], "3rd+")
    key = (o, band)
    births[key] = births.get(key, 0) + pd.Series([number(x) for x in r[3:]], index=years).fillna(0)
B = {o: pd.DataFrame({b: births[(o, b)] for b in AGES}) for o in ORD}
W = women()
tfr_off = national_rates()["합계출산율"]
yrs = [y for y in tfr_off.index if y in W.index and y >= 1993]
raw = pd.DataFrame({o: (B[o].loc[yrs] / W.loc[yrs, AGES] * 5).sum(axis=1) for o in ORD})
kr_tfr = raw.mul(tfr_off[yrs] / raw.sum(axis=1), axis=0)  # rescaled to the official TFR
kr_mean = pd.DataFrame({o: (B[o].loc[yrs] * MIDS).sum(axis=1) / B[o].loc[yrs].sum(axis=1) for o in ORD})
allb = sum(B[o] for o in ORD).loc[yrs]
kr_mean_all = (allb * MIDS).sum(axis=1) / allb.sum(axis=1)
kr_adj, kr_r = adjust(kr_tfr, kr_mean)
KR = pd.DataFrame({"tfr": tfr_off[yrs], "adj_by_order": kr_adj, "adj_total": tfr_off[yrs] / (1 - r_of(kr_mean_all)),
                   "mean_age_all_mid": kr_mean_all, **{f"r_{o}": kr_r[o] for o in ORD}})

# --- Japan: official TFR by order (4-18) and official mean age (4-19) ----------------------------
official = JC.by_year_rows("2024_4-19.csv", ["all", "1st", "2nd", "3rd"]).rename(columns={"3rd": "3rd+"})  # the table has 3rd only
jp_tfr = pd.DataFrame({o: JC.tfr_order[o].sum(axis=1) for o in ORD}).loc[2009:2024]
jp_adj, jp_r = adjust(jp_tfr, official.loc[2009:2024])
tot = JC.basic.tfr.loc[1995:2024]
JP = pd.DataFrame({"tfr": tot, "adj_total": tot / (1 - r_of(official.loc[1995:2024, "all"]))})
JP["adj_by_order"] = jp_adj

# --- check: band-midpoint mean age against the official one, Japan 4-17 vs 4-19 ----------------
s417 = JC.sections("2024_4-17.csv", 6)
blocks = {"第　１　子": "1st", "第　２　子": "2nd", "第３子以上": "3rd+", "総　　　数": "all"}
bands417 = ["19歳以下", "20～24歳", "25～29歳", "30～34歳", "35～39歳", "40～44歳", "45歳以上"]
mid = pd.DataFrame({o: sum(s417[(b, a)] * MIDS[i] for i, a in enumerate(bands417)) / sum(s417[(b, a)] for a in bands417) for b, o in blocks.items()}).loc[2009:2024]
CHK = pd.DataFrame({o: dict(level_gap=(mid[o] - official.loc[2009:2024, o]).mean(),
                            r_mid=r_of(mid[o]).loc[2010:2023].mean(), r_official=r_of(official.loc[2009:2024, o]).loc[2010:2023].mean(),
                            r_corr=r_of(mid[o]).loc[2010:2023].corr(r_of(official.loc[2009:2024, o]).loc[2010:2023])) for o in ("all", "1st", "2nd")}).T

for name, frame in (("korea", KR), ("japan", JP), ("midpoint_check", CHK)):
    frame.to_csv(REP / f"24_{name}.csv")
assert np.allclose(kr_tfr.sum(axis=1), tfr_off[yrs]), "rescaling"

pd.set_option("display.width", 200)
print("check: band-midpoint mean age vs official (Japan 2009~2024); r = yearly change\n", CHK.round(3).to_string())
for label, F, show in (("Korea", KR, [2005, 2010, 2015, 2019, 2020, 2021, 2022, 2023, 2024, 2025]), ("Japan", JP, [2000, 2003, 2005, 2007, 2009, 2010, 2012, 2015, 2017, 2019, 2021, 2023, 2024])):
    out = F.loc[[y for y in show if y in F.index]].copy()
    out["adj_total_3y"] = F.adj_total.rolling(3, center=True).mean()
    print(f"\n{label}: observed and tempo-adjusted TFR\n", out.round(3).to_string())
