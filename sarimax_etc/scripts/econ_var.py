"""E14: VAR of mortgage rate, real house prices, employment 25~29, marriages and births (12-month differences); report only.

Rules are in notes/decisions.md. Run from births/: python -I scripts/econ_var.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))  # appended last: scripts run with python -I
from common import RAW, REP  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from statsmodels.tsa.api import VAR  # noqa: E402

KOSIS = RAW / "kosis_api_20261110"
ECOS = RAW / "ecos_20261110" / "ecos_monthly.csv"
ORDER = ["mortgage_rate", "real_house_price", "emp_25_29", "marriages", "births"]
ECON = ORDER[:3]
TRAIN = ("2009-01-01", "2023-12-01")
FUTURE = pd.date_range("2024-01-01", "2026-08-01", freq="MS")


def months(s):
    return pd.to_datetime(s.astype(str), format="%Y%m")


def vital(kind, region="00"):
    """National monthly count from DT_1B8000G by its Korean name, e.g. '출생아수(명)', '혼인건수(건)', '이혼건수(건)'."""
    d = pd.read_csv(KOSIS / "vital_events_monthly_by_province.csv", dtype=str)
    d = d[(d.C1 == region) & (d.C2_NM == kind)]
    s = pd.Series(d.DT.astype(float).values, index=months(d.PRD_DE)).sort_index().asfreq("MS")
    assert s.notna().all() and len(s) > 500, kind
    return s


def ecos(series, item=None):
    d = pd.read_csv(ECOS, dtype={"month": str, "item": str})
    d = d[d.series == series]
    if item:
        d = d[d.item_name == item]
    s = pd.Series(d.value.astype(float).values, index=months(d.month)).sort_index()
    assert s.index.is_unique, series
    return s.asfreq("MS")


def employment(age="25 - 29세"):
    d = pd.read_csv(KOSIS / "labor_force_by_sex_age_monthly.csv", dtype=str)
    d = d[(d.C1_NM == "계") & (d.C2_NM == age) & (d.ITM_NM == "고용률")]
    return pd.Series(d.DT.astype(float).values, index=months(d.PRD_DE)).sort_index().asfreq("MS")


def levels():
    L = pd.DataFrame({
        "mortgage_rate": ecos("loan_rate_mortgage_new"),
        "real_house_price": np.log(ecos("house_price_kb_national")) - np.log(ecos("cpi")),
        "emp_25_29": employment(),
        "marriages": np.log(vital("혼인건수(건)")),
        "births": np.log(vital("출생아수(명)")),
    })
    return L["2008-01-01":"2026-08-01"]


def main(rule="aic"):
    """rule = aic (declared) or bic (post-hoc sensitivity: AIC hit the 13-lag cap); bic writes 26b_*."""
    L = levels()
    assert L.notna().all().all(), L.isna().sum()
    Z = L.diff(12).dropna()[ORDER]
    tr = Z[TRAIN[0]:TRAIN[1]]
    sel = VAR(tr).select_order(13)
    p = max(int(getattr(sel, rule)), 1)
    pre = "26_" if rule == "aic" else "26b_"
    r = VAR(tr).fit(p)

    gr = []
    for caused, causing in (("marriages", ECON), ("births", ECON), ("births", ["marriages"]), ("marriages", ["births"])):
        t = r.test_causality(caused, causing, kind="f")
        gr.append(dict(caused=caused, causing="+".join(causing), F=t.test_statistic, p=t.pvalue, df=str(t.df)))
    G = pd.DataFrame(gr)

    irf = r.irf(36).orth_irfs  # [h, response, shock]
    I = pd.DataFrame([dict(h=h, response=ORDER[i], shock=ORDER[j], value=irf[h, i, j]) for h in range(37) for i in (3, 4) for j in range(5)])
    fe = r.fevd(36).decomp  # [response, h, shock]
    F = pd.DataFrame([dict(response=ORDER[i], h=h, **{s: fe[i, h - 1, j] for j, s in enumerate(ORDER)}) for i in (3, 4) for h in (12, 24, 36)])

    # conditional forecast: iterate the reduced form; econ variables either forecast or set to their actual path
    A, c = r.coefs, r.intercept
    def run(actual_econ):
        hist = Z[:TRAIN[1]].copy()
        for t in FUTURE:
            x = c + sum(A[i] @ hist.iloc[-1 - i].values for i in range(p))
            if actual_econ:
                x[:3] = Z.loc[t, ECON].values
            hist.loc[t] = x
        return hist.loc[FUTURE]
    out = {}
    for name, g in (("unconditional", run(False)), ("actual economy", run(True)), ("actual", Z.loc[FUTURE])):
        lv = L[["marriages", "births"]][:TRAIN[1]].copy()
        for t in FUTURE:
            lv.loc[t] = lv.loc[t - pd.DateOffset(months=12)].values + g.loc[t, ["marriages", "births"]].values
        out[name] = np.exp(lv.loc[FUTURE])
    assert np.allclose(out["actual"].values, np.exp(L.loc[FUTURE, ["marriages", "births"]]).values), "level rebuild"
    C = pd.concat(out, axis=1)
    C.to_csv(REP / (pre + "conditional_paths.csv"))
    tot = {k: v.groupby(v.index.year).sum() for k, v in out.items()}
    rows = []
    for var in ("marriages", "births"):
        for yr in (2024, 2025, 2026):
            a, u, e = tot["actual"].loc[yr, var], tot["unconditional"].loc[yr, var], tot["actual economy"].loc[yr, var]
            rows.append(dict(var=var, year=yr, actual=a, unconditional=u, actual_economy=e, gap=a - u, explained_share=(e - u) / (a - u)))
    S = pd.DataFrame(rows)

    G.to_csv(REP / (pre + "granger.csv"), index=False)
    I.to_csv(REP / (pre + "irf.csv"), index=False)
    F.to_csv(REP / (pre + "fevd.csv"), index=False)
    S.to_csv(REP / (pre + "explained_share.csv"), index=False)
    pd.set_option("display.width", 200)
    print(f"VAR on 12-month differences, {TRAIN[0][:7]}~{TRAIN[1][:7]}, n={len(tr)}, lag p={p} (AIC; BIC={sel.bic})")
    print("\nGranger (overlapping differences: weak evidence)\n", G.round(4).to_string(index=False))
    print("\nFEVD share of forecast-error variance\n", F.round(3).to_string(index=False))
    peak = I[I.shock.isin(ECON)].groupby(["response", "shock"]).value.agg(["min", "max"])
    print("\northogonal IRF range over 36 months (log points of 12-month change)\n", peak.round(4).to_string())
    print("\nyearly totals (2026 = Jan~Aug); explained_share = (actual economy - unconditional) / (actual - unconditional)\n", S.round(3).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "aic")
