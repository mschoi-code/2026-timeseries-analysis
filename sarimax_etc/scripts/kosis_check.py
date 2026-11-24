"""Check the 2026-11-07 KOSIS downloads and decompose births into population and rate effects; report only.

Run from births/: python -I scripts/kosis_check.py
"""
import re
import sys
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))  # appended last: scripts run with python -I
sys.path.append(str(ROOT / "scripts"))
from common import REP, RAW, load_monthly  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from realtime import scenarios  # noqa: E402

pd.set_option("display.width", 170)
NEW = RAW / "kosis_20261107"
M = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
R_ID = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
AGES = ["15 - 19세", "20 - 24세", "25 - 29세", "30 - 34세", "35 - 39세", "40 - 44세", "45 - 49세"]
MARR_AGES = AGES + ["50 - 54세"]


def col_index(ref):
    n = 0
    for ch in re.match(r"[A-Z]+", ref).group():
        n = n * 26 + ord(ch) - 64
    return n - 1


def sheet(pattern):
    """Rows of the data sheet of the one file matching pattern (openpyxl is not in the school env)."""
    paths = list(NEW.glob(pattern))
    assert len(paths) == 1, (pattern, paths)
    with zipfile.ZipFile(paths[0]) as z:
        strings = ["".join(t.text or "" for t in si.iter(M + "t")) for si in ET.fromstring(z.read("xl/sharedStrings.xml"))]
        book = ET.fromstring(z.read("xl/workbook.xml"))
        rels = {r.attrib["Id"]: r.attrib["Target"] for r in ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))}
        first = book.find(M + "sheets")[0]
        assert first.attrib["name"] == "데이터", first.attrib["name"]
        target = rels[first.attrib[R_ID]].lstrip("/")
        target = target if target.startswith("xl/") else "xl/" + target
        rows = []
        for row in ET.fromstring(z.read(target)).find(M + "sheetData"):
            vals = []
            for c in row:
                i = col_index(c.attrib["r"])
                vals += [None] * (i + 1 - len(vals))
                v = c.find(M + "v")
                if v is not None:
                    vals[i] = strings[int(v.text)] if c.attrib.get("t") == "s" else v.text
            rows.append(vals)
    return rows


def number(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def annual_item(pattern, label):
    rows = sheet(pattern)
    row = next(r for r in rows if r[0] == label)
    return pd.Series([number(x) for x in row[1:]], index=[int(y) for y in rows[0][1:]])


def national_rates():
    """Age-specific fertility per 1,000 women: years as rows, 5-year age bands as columns."""
    rows = sheet("시도_합계출산율*")
    years, items = rows[0], rows[1]
    nat = next(r for r in rows if r[0] == "전국")
    out = {}
    for j in range(1, len(nat)):
        name = items[j].replace("모의 연령별출산율:", "")
        out.setdefault(int(years[j]), {})[name] = number(nat[j])
    frame = pd.DataFrame(out).T.sort_index()
    return frame.rename(columns=lambda c: c if c == "합계출산율" else c.replace("-", " - "))


def women():
    """Women by 5-year band from the team file; estimates through 2022, medium projection after."""
    p = pd.read_csv(RAW / "성_및_연령별_추계인구_1세별_5세별_전국.csv", header=None, dtype=str)
    years = p.iloc[2, 1:].astype(int).values
    body = p.iloc[3:].set_index(0)
    return pd.DataFrame({a: pd.to_numeric(body.loc[a], errors="coerce").values for a in MARR_AGES}, index=years)


def kitagawa(r0, r1, w0, w1):
    """Population and rate effects with average weights; r per 1,000, by band."""
    return pd.DataFrame({"population": (w1 - w0) * (r0 + r1) / 2 / 1000, "rate": (r1 - r0) * (w0 + w1) / 2 / 1000})


def main():
    m = load_monthly()
    W = women()
    R = national_rates()
    births = annual_item("출생아수__*", "출생아수(명)")
    marr = annual_item("혼인건수__*", "혼인건수(건)")
    team = m.groupby(m.index.year).sum()

    print("A. cross-checks (kosis minus team file)")
    yrs = [y for y in births.index if 1981 <= y <= 2025]
    print("   annual births 1981~2025: nonzero", int((births[yrs] != team.births[yrs]).sum()), "max abs", (births[yrs] - team.births[yrs]).abs().max())
    print("   annual marriages 1981~2025: nonzero", int((marr[yrs] != team.marr[yrs]).sum()), "max abs", (marr[yrs] - team.marr[yrs]).abs().max())
    rows = sheet("시도_시군구_월별_혼인*")
    nat = next(r for r in rows if r[0] == "전국")
    mm = pd.Series([number(x) for x in nat[1:]], index=pd.to_datetime(rows[0][1:], format="%Y.%m"))
    print(f"   monthly marriages {mm.index[0]:%Y-%m}~{mm.index[-1]:%Y-%m}: nonzero", int((mm != m.marr[mm.index]).sum()), "max abs", (mm - m.marr[mm.index]).abs().max())

    order_rows = sheet("성_모의*")
    head = order_rows[0]
    sex = age = None
    order = {}
    for r in order_rows[1:]:
        r = r + [None] * (3 - len(r))
        sex, age = r[0] or sex, r[1] or age
        order[(sex, age, r[2])] = pd.Series([number(x) for x in r[3:]], index=[int(y) for y in head[3:]])
    total = order[("계", "계", "총계")]
    print("   birth-order table total vs annual births: nonzero", int((total[yrs] != births[yrs]).sum()), "max abs", (total[yrs] - births[yrs]).abs().max())

    raw = (NEW / "101_DT_1B83A02_20261107182853.xls").read_bytes().decode("euc-kr", "replace")  # SpreadsheetML with a stray "<" in the metadata sheet
    grid = [re.findall(r"<Data[^>]*>(.*?)</Data>", row, flags=re.S) for row in re.findall(r"<Row[^>]*>(.*?)</Row>", raw, flags=re.S)]
    couple_head = next(r for r in grid if r[:2] == ["남편의 연령별", "아내의 연령별"])
    couple = next(r for r in grid if r[:2] == ["계", "계"])
    pair = pd.Series([number(x) for x in couple[4:]], index=[int(y.split()[0]) for y in couple_head[4:]])
    print(f"   couple-age table total {pair.index[0]}~{pair.index[-1]} vs annual marriages: nonzero", int((pair != marr[pair.index]).sum()))

    rebuilt = R[AGES].sum(axis=1) * 5 / 1000
    print("   TFR rebuilt from bands, max abs gap", round((rebuilt - R["합계출산율"]).abs().max(), 4))
    # up to 0.0135 in other years, smooth over time: bands and the official single-year sum differ, not rounding alone
    assert (rebuilt - R["합계출산율"]).loc[[2015, 2023, 2025]].abs().max() < 0.005  # the years decomposed below

    print("\nB. denominator check: rates x team-file women vs actual births")
    implied = pd.Series({y: (R.loc[y, AGES] * W.loc[y, AGES]).sum() / 1000 for y in R.index})
    gap = implied / births[R.index] - 1
    gap.rename("gap").to_csv(REP / "10_denominator_gap.csv")
    print((gap.loc[[1993, 2000, 2005, 2010, 2015, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]] * 100).round(2).to_string())

    print("\nC. Kitagawa decomposition of births")
    frames = []
    for a, b in ((2015, 2023), (2023, 2025)):
        e = kitagawa(R.loc[a, AGES], R.loc[b, AGES], W.loc[a, AGES], W.loc[b, AGES])
        e["from"], e["to"] = a, b
        frames.append(e)
        impl, act = e[["population", "rate"]].sum().sum(), births[b] - births[a]
        print(f"   {a}->{b}: actual {act:+,.0f}, implied {impl:+,.0f}; population {e.population.sum():+,.0f} ({e.population.sum() / impl:.1%}), rate {e.rate.sum():+,.0f} ({e.rate.sum() / impl:.1%})")
        print(e[["population", "rate"]].round(0).to_string())
    pd.concat(frames).to_csv(REP / "10_kitagawa_births.csv")
    print("   births 2023->2025 %+.2f%%, TFR %+.2f%%" % ((births[2025] / births[2023] - 1) * 100, (R.loc[2025, "합계출산율"] / R.loc[2023, "합계출산율"] - 1) * 100))

    print("\nD. Kitagawa decomposition of marriages by wife's age")
    rows = sheet("시도_성_연령별*")
    yr_head, item = rows[0], rows[1]
    wife = {}
    region = None
    for r in rows[2:]:
        region = r[0] or region  # the region label appears only on the first row of each block
        if region == "계" and r[1] in MARR_AGES:
            for j in range(2, len(r)):
                if item[j].startswith("아내"):
                    wife.setdefault(int(yr_head[j]), {})[r[1]] = number(r[j])
    MR = pd.DataFrame(wife).T.sort_index()[MARR_AGES]
    for yr in (2023, 2025):
        print(f"   {yr} implied (15~54) {(MR.loc[yr] * W.loc[yr]).sum() / 1000:,.0f}, actual {marr[yr]:,.0f}")
    e = kitagawa(MR.loc[2023], MR.loc[2025], W.loc[2023], W.loc[2025])
    e.to_csv(REP / "10_kitagawa_marriages.csv")
    impl = e.sum().sum()
    print(f"   2023->2025 actual {marr[2025] - marr[2023]:+,.0f}, implied {impl:+,.0f}; population {e.population.sum():+,.0f} ({e.population.sum() / impl:.1%}), rate {e.rate.sum():+,.0f} ({e.rate.sum() / impl:.1%})")
    print(e.round(0).to_string())

    print("\nE. birth order")
    third = sum(order[("계", "계", k)] for k in ("3아", "4아", "5아", "6아", "7아", "8아 이상"))
    T = pd.DataFrame({"births": total, "first": order[("계", "계", "1아")], "second": order[("계", "계", "2아")], "third_plus": third, "marriages": marr})
    T["first_share"] = T["first"] / T.births
    T["first_per_prev_marr"] = T["first"] / T.marriages.shift(1)
    T = T.loc[2015:2025]
    T.to_csv(REP / "10_birth_order.csv")
    print(T.loc[2019:].round(3).to_string())
    for a, b in ((2023, 2024), (2024, 2025), (2023, 2025)):
        d = T.loc[b] - T.loc[a]
        print(f"   {a}->{b}: births {d.births:+,.0f}; first {d['first']:+,.0f} ({d['first'] / d.births:.1%}), second {d.second:+,.0f} ({d.second / d.births:.1%}), third+ {d.third_plus:+,.0f}")
    by_age = pd.Series({a: order[("계", a, "1아")][2025] - order[("계", a, "1아")][2023] for a in AGES})
    print("   first births change 2023->2025 by mother's age:", by_age.round(0).to_dict())

    print("\nF. structural effect: fixed 2025 rates against the scenario model")
    fixed = pd.Series({y: (R.loc[2025, AGES] * W.loc[y, AGES]).sum() / 1000 for y in range(2025, 2036)})
    fixed.rename("births_at_2025_rates").to_csv(REP / "10_fixed_rate_path.csv")
    acct = fixed[2031] / fixed[2026] - 1
    print(f"   fixed rates 2026->2031 {acct:+.2%}, peak {fixed.idxmax()}, 2035 vs 2026 {fixed[2035] / fixed[2026] - 1:+.2%}")
    w3034 = np.log(W.loc[2031, "30 - 34세"] / W.loc[2026, "30 - 34세"])
    w2534 = W["25 - 29세"] + W["30 - 34세"]
    print(f"   women 30~34 2026->2031 {np.exp(w3034) - 1:+.2%}, women 25~34 {w2534[2031] / w2534[2026] - 1:+.2%}")
    print("   all bands 2026->2031:", (W.loc[2031, AGES] / W.loc[2026, AGES] - 1).round(3).to_dict())
    pop = w2534.copy()
    pop.index = pd.to_datetime([f"{y}-07-01" for y in pop.index])
    pop = pop.reindex(pd.date_range(pop.index[0], pop.index[-1], freq="MS")).interpolate("linear")
    last = m.index[-1]
    out = []
    for lo, hi in ((9, 32), (12, 35)):
        sc, co = scenarios(lo, hi)
        for _, c in co.iterrows():
            s = sc[(sc.order == c.order) & (sc.window == c.window) & (sc.rate == "hold") & (sc.year == 2031)].set_index("population").births
            months = pd.date_range("2031-01-01", "2031-12-01", freq="MS")
            ratio = np.mean([np.mean([pop[min(t - pd.DateOffset(months=k), pd.Timestamp("2035-07-01"))] / pop[last] if t - pd.DateOffset(months=k) > last else 1.0 for k in range(lo, hi + 1)]) for t in months])
            women_part, marr_part = c.b_women * w3034, c.b_marr * np.log(ratio)
            out.append(dict(order=c.order, window=c.window, model_pct=(s["projected"] / s["frozen"] - 1) * 100, b_women=c.b_women, b_marr=c.b_marr,
                            coef_sum=c.b_women + c.b_marr, women_part=women_part * 100, marr_part=marr_part * 100, predicted_pct=(np.exp(women_part + marr_part) - 1) * 100,
                            structure_people=s["projected"] - s["frozen"], accounting_pct=acct * 100, ratio_to_accounting=(s["projected"] / s["frozen"] - 1) / acct))
    F = pd.DataFrame(out)
    F.to_csv(REP / "10_structural_comparison.csv", index=False)
    print(F.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
