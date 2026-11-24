"""E16: where the 2023->2025 birth rise sits: birth order, marriage duration, non-marital births, divorces; report only.

Rules are in notes/decisions.md. Run from births/: python -I scripts/decompose_rebound.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))  # appended last: scripts run with python -I
sys.path.append(str(ROOT / "scripts"))
from common import REP  # noqa: E402
import pandas as pd  # noqa: E402
from econ_var import KOSIS, vital  # noqa: E402
from kosis_check import number, sheet  # noqa: E402

YEARS = list(range(2015, 2026))


def share_of_rise(T, a=2023, b=2025):
    d = T.loc[b] - T.loc[a]
    return pd.DataFrame({f"{a}": T.loc[a], f"{b}": T.loc[b], "change": d, "share_of_rise_pct": d / d.sum() * 100})


def by_order():
    rows = sheet("성_모의*")
    years = [int(y) for y in rows[0][3:]]
    assert rows[1][:3] == ["계", "계", "총계"], rows[1][:3]
    out = {}
    for r in rows[2:]:
        if r[0] or r[1]:  # the next sex or age block starts: national total block is over
            break
        o = {"1아": "1st", "2아": "2nd", "미상": "unknown"}.get(r[2], "3rd+")
        out[o] = out.get(o, 0) + pd.Series([number(x) for x in r[3:]], index=years)
    T = pd.DataFrame(out).loc[YEARS]
    total = pd.Series([number(x) for x in rows[1][3:]], index=years).loc[YEARS]
    assert (T.sum(axis=1) - total).abs().max() < 1, "orders do not add up"
    return T


def by_duration():
    d = pd.read_csv(KOSIS / "births_by_marriage_duration.csv", dtype=str)
    d = d[(d.C1 == "00") & (d.C2_NM == "계")]
    W = d.pivot(index="PRD_DE", columns="C3_NM", values="DT").astype(float)
    W.index = W.index.astype(int)
    groups = {"<1y": ["1년미만"], "1y": ["1년"], "2y": ["2년"], "3~4y": ["3년", "4년"],
              "5y+": ["5년", "6년", "7년", "8년", "9년", "10~14년", "15~19년", "20년이상"], "unknown": ["미상"]}
    T = pd.DataFrame({g: W[c].sum(axis=1) for g, c in groups.items()}).loc[YEARS]
    assert (T.sum(axis=1) - W["계"].loc[YEARS]).abs().max() < 1
    return T


def non_marital():
    d = pd.read_csv(KOSIS / "births_by_legal_marital_status.csv", dtype=str)
    W = d[d.C1 == "00"].pivot(index="PRD_DE", columns="ITM_NM", values="DT").astype(float)
    W.index = W.index.astype(int)
    W = W.loc[YEARS]
    return pd.DataFrame({"births": W["총계"], "non_marital": W["혼인외의 자"], "non_marital_pct": W["혼인외의 자"] / W["총계"] * 100})


def divorces():
    dv, mr = vital("이혼건수(건)"), vital("혼인건수(건)")
    A = pd.DataFrame({"divorces": dv.groupby(dv.index.year).sum(), "marriages": mr.groupby(mr.index.year).sum()}).loc[YEARS]
    A["divorce_yoy_pct"] = A.divorces.pct_change() * 100
    A["divorces_per_marriage"] = A.divorces / A.marriages
    last = dv.index[-1]
    ytd = lambda s, y: s[f"{y}-01-01":f"{y}-{last.month:02d}-01"].sum()
    A.loc[f"{last.year} Jan~{last.month}"] = [ytd(dv, last.year), ytd(mr, last.year), (ytd(dv, last.year) / ytd(dv, last.year - 1) - 1) * 100, ytd(dv, last.year) / ytd(mr, last.year)]
    return A


def main():
    O, D, N, V = by_order(), by_duration(), non_marital(), divorces()
    base = D.loc[2015:2019].mean()
    Dpct = (base / base.sum() * 100).rename("share_2015_2019_pct")
    for name, frame in (("order", O), ("duration", D), ("non_marital", N), ("divorces", V)):
        frame.to_csv(REP / f"28_{name}.csv")
    pd.concat({"order": share_of_rise(O), "duration": share_of_rise(D)}).to_csv(REP / "28_share_of_rise.csv")
    pd.set_option("display.width", 200)
    print("births by order\n", O.astype(int).to_string(), "\n\n2023->2025 rise by order\n", share_of_rise(O).round(1).to_string())
    print("\nbirths by marriage duration\n", D.astype(int).to_string(), "\n\n2023->2025 rise by duration\n", share_of_rise(D).round(1).to_string())
    print("\n2015~2019 duration distribution (1y + 2y ~ the 12~35-month window)\n", Dpct.round(1).to_string())
    print("\nnon-marital births\n", N.round(2).to_string())
    print("\ndivorces\n", V.round(3).to_string())


if __name__ == "__main__":
    main()
