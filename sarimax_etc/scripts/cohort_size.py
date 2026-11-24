"""E18: does a cohort's own size (Easterlin) or its parents' marriage rate (transmission) move its age-specific marriage rate? Report only.

Rules are in notes/decisions.md. Run from sarimax_etc/: python -I scripts/cohort_size.py
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
import statsmodels.api as sm  # noqa: E402
from kosis_check import annual_item, women  # noqa: E402
from province_standardized import wife_rates  # noqa: E402

warnings.filterwarnings("ignore")
LO = {"20 - 24세": 20, "25 - 29세": 25, "30 - 34세": 30, "35 - 39세": 35}
POP_BANDS = ["25 - 29세", "30 - 34세", "35 - 39세"]  # 20~24 would need women aged 10~14, not in the team file
PARENT_GAP = 2  # parents marry about two years before the birth


def mean_over(s, first, last):
    """Mean of an annual series over first..last, NaN if any year is missing."""
    yrs = range(first, last + 1)
    return s.reindex(yrs).mean() if s.reindex(yrs).notna().all() else np.nan


def panel(who):
    rates = wife_rates(who)["전국"]
    W = women()
    births = annual_item("출생아수__합계출산율*", "출생아수(명)")
    cmr = annual_item("혼인건수__조혼인율*", "조혼인율(천명당)")
    rows = []
    for band, lo in LO.items():
        younger = f"{lo - 10} - {lo - 6}세"
        for t in rates.index:
            first, last = t - lo - 4, t - lo  # birth years of the band in year t
            rows.append(dict(band=band, year=t, cohort=t - lo, log_rate=np.log(rates.loc[t, band]),
                             size_pop=np.log(W.loc[t - 10, younger]) if younger in W.columns and t - 10 in W.index else np.nan,
                             size_birth=np.log(mean_over(births, first, last)),
                             parent_rate=np.log(mean_over(cmr, first - PARENT_GAP, last - PARENT_GAP))))
    return pd.DataFrame(rows)


def fe_fit(P, xs, bands):
    d = P[P.band.isin(bands)].dropna(subset=["log_rate", *xs])
    F = pd.get_dummies(d[["band", "year"]].astype(str), drop_first=True, dtype=float)
    X = sm.add_constant(pd.concat([d[xs], F], axis=1))
    r = sm.OLS(d.log_rate, X).fit(cov_type="cluster", cov_kwds={"groups": d.cohort.values}, use_t=True)
    return d, r


def main():
    assert women().columns[0] == "15 - 19세"
    out, raw = [], []
    models = {"M1 size_pop": (["size_pop"], POP_BANDS), "M2 size_birth": (["size_birth"], list(LO)),
              "M3 parent_rate": (["parent_rate"], list(LO)), "M4 size_birth + parent_rate": (["size_birth", "parent_rate"], list(LO))}
    for who in ("아내", "남편"):
        P = panel(who)
        assert P.log_rate.notna().all() and len(P) == 4 * 36, (who, len(P))
        for name, (xs, bands) in models.items():
            if who == "남편" and name != "M1 size_pop":
                continue  # husband rates: declared sensitivity for M1 only (cohort size is women's)
            d, r = fe_fit(P, xs, bands)
            for x in xs:
                out.append(dict(rates=who, model=name, term=x, b=r.params[x], se=r.bse[x], p=r.pvalues[x], n=len(d),
                                years=f"{d.year.min()}~{d.year.max()}", clusters=d.cohort.nunique(), bands=len(bands)))
        if who == "아내":
            P.to_csv(REP / "30_panel.csv", index=False)
            for band, g in P.groupby("band"):
                for x in ("size_pop", "size_birth", "parent_rate"):
                    g2 = g.dropna(subset=[x])
                    if len(g2) > 5:
                        raw.append(dict(band=band, x=x, n=len(g2), corr_level=g2.log_rate.corr(g2[x]),
                                        corr_diff=g2.log_rate.diff().corr(g2[x].diff())))
            both = P.dropna(subset=["size_birth", "parent_rate"])
            print(f"corr(size_birth, parent_rate) on M4 sample: {both.size_birth.corr(both.parent_rate):.3f}")
    T, Rw = pd.DataFrame(out), pd.DataFrame(raw)
    T.to_csv(REP / "30_fe_models.csv", index=False)
    Rw.to_csv(REP / "30_raw_correlations.csv", index=False)
    pd.set_option("display.width", 200)
    print(T.round(3).to_string(index=False))
    print("\nraw correlations, no fixed effects (descriptive; level correlations of trending series are not evidence)")
    print(Rw.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
