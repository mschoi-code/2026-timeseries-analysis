"""E7: COVID marriage deficit and its cumulative catch-up against pre-2020 counterfactuals; report only.

Rules are in notes/decisions.md. Run from births/: python -I scripts/covid_catchup.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))  # appended last: scripts run with python -I
sys.path.append(str(ROOT / "scripts"))
from common import REP, load_monthly  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from population_variable import POP_2534  # noqa: E402

m = load_monthly().marr
END = m.index[-1]
CFS = {"A count 2010~2019": ("count", 2010), "B count 2015~2019": ("count", 2015),
       "C rate 2010~2019": ("rate", 2010), "D hold 2019": ("hold", None)}


def counterfactual(kind, first, last, until=END):
    """Monthly marriages from first-01 to until, fitted on first..last only."""
    t = pd.date_range(f"{first}-01-01", until, freq="MS")
    if kind == "hold":
        same = m[str(last)]
        return pd.Series(same.reindex([pd.Timestamp(last, d.month, 1) for d in t]).values, index=t)
    pop = POP_2534.reindex(t) if kind == "rate" else pd.Series(1.0, index=t)
    X = np.column_stack([np.ones(len(t)), np.arange(len(t)) / 12] + [(t.month == k).astype(float) for k in range(2, 13)])
    fit = t.year <= last
    y = np.log(m.reindex(t)[fit] / pop[fit]).values
    b, *_ = np.linalg.lstsq(X[fit], y, rcond=None)
    smear = np.exp(y - X[fit] @ b).mean()  # Duan: cumulative sums would carry the median bias
    out = pd.Series(np.exp(X @ b) * smear * pop.values, index=t)
    assert abs(out[fit].sum() / m.reindex(t)[fit].sum() - 1) < 0.01, "in-sample sum off"
    return out


rows, annual = [], []
for name, (kind, first) in CFS.items():
    full = counterfactual(kind, first or 2019, 2019)
    cf = full["2020":]
    gap = m["2020":] - cf
    cum = gap.cumsum()
    repay = np.clip(np.minimum(gap, -cum.shift(1, fill_value=0)), 0, None)  # positive gap that fills the open backlog
    repaid = repay.groupby(repay.index.year).sum()
    deficit = gap[:"2022"].sum()
    surplus = gap["2023":].sum()
    back = cum[cum.idxmin():]
    back = back[back >= 0]
    vfirst = (first - 3) if first else None
    vcf = counterfactual(kind, vfirst or 2016, 2016, "2019-12-01")["2017":]
    yearly = gap.groupby(gap.index.year).sum()
    rows.append(dict(cf=name, deficit_2020_22=deficit, min_cum=cum.min(), min_month=cum.idxmin().strftime("%Y-%m"),
                     back_to_zero=back.index[0].strftime("%Y-%m") if len(back) else "not yet",
                     cum_end=cum.iloc[-1], surplus_2023_on=surplus,
                     repay_share=min(surplus, -deficit) / surplus * 100 if surplus > 0 else np.nan,
                     valid_2017_19_pct=(m["2017":"2019"].sum() / vcf.sum() - 1) * 100,
                     gap_2022=yearly[2022], gap_2024=yearly[2024],
                     fit_2019_pct=(full["2019"].sum() / m["2019"].sum() - 1) * 100))
    for yr, v in yearly.items():
        annual.append(dict(cf=name, year=yr, actual=m[str(yr)].sum(), counterfactual=cf[str(yr)].sum(), gap=v, repaid=repaid[yr]))

order = pd.read_csv(REP / "10_birth_order.csv", index_col=0)
r = order.loc[2023:2025, "first_per_prev_marr"].mean()
S = pd.DataFrame(rows)
S["first_births_from_gap_23_25"] = r * (S.gap_2024 - S.gap_2022)
S["share_of_births_rise"] = S.first_births_from_gap_23_25 / (order.loc[2025, "births"] - order.loc[2023, "births"]) * 100
A = pd.DataFrame(annual)
S.to_csv(REP / "19_catchup_summary.csv", index=False)
A.to_csv(REP / "19_catchup_annual.csv", index=False)

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 20)
print(f"r = {r:.3f}, data end {END:%Y-%m}")
print(A.pivot(index="year", columns="cf", values="gap").round(0))
print(A[A.cf == S.cf[0]][["year", "actual"]].set_index("year").T)
print(A.pivot(index="year", columns="cf", values="counterfactual").round(0))
print("repaid (marriages filling the backlog), first births a year later = r x repaid")
print(A.pivot(index="year", columns="cf", values="repaid").round(0))
print(S.round(1).T)
