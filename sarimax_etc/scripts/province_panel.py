"""Province panel: do a province's marriages lead its fertility once national shocks are removed? Report only.

Rules are in notes/decisions.md. Run from sarimax_etc/: python -I scripts/province_panel.py
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
from kosis_check import number, sheet  # noqa: E402

warnings.filterwarnings("ignore")
pd.set_option("display.width", 180)
pd.set_option("display.max_columns", 30)
NATIONAL, SEJONG, CHUNGNAM = "전국", "세종특별자치시", "충청남도"
PRIMARY = (1, 2)  # lags closest to the proposal's 12~35 month window
RI_DRAWS = 1000


def annual_pairs(pattern, wanted):
    """{item: years x provinces} from a table with years and items in its two header rows."""
    rows = sheet(pattern)
    years, items = rows[0], rows[1]
    out = {w: {} for w in wanted}
    for row in rows[2:]:
        for j in range(1, len(row)):
            if items[j] in out:
                out[items[j]].setdefault(row[0], {})[int(years[j])] = number(row[j])
    return {k: pd.DataFrame(v).sort_index() for k, v in out.items()}


def load():
    table = annual_pairs("시군구_출생아수*", ("출생아수", "합계출산율"))
    rate = annual_pairs("시도_일반혼인율*", ("아내",))["아내"]
    rows = sheet("시도_시군구_월별_혼인*")
    monthly = pd.DataFrame({r[0]: [number(x) for x in r[1:]] for r in rows[1:]}, index=pd.to_datetime(rows[0][1:], format="%Y.%m"))
    return table["출생아수"], table["합계출산율"], rate, monthly


def design(tfr, rate, lags, start, end, diff=False, trends=False, drop=None):
    """Long-form y and X with year effects, and province effects unless differenced."""
    units = [c for c in tfr.columns if c not in (NATIONAL, SEJONG, drop)]
    y_wide, g = np.log(tfr[units]), np.log(rate[units]).reindex(range(1985, 2030))
    if diff:
        y_wide, g = y_wide.diff(), g.diff()
    yrs = list(range(start, end + 1))
    y = y_wide.loc[yrs].stack()
    y.index.names = ["year", "unit"]
    X = pd.DataFrame({f"L{L}": g.shift(L).loc[yrs].stack() for L in lags}).reindex(y.index)
    assert X.notna().all().all() and y.notna().all() and len(y) == len(units) * len(yrs), "panel must be balanced"
    year = y.index.get_level_values("year")
    unit = y.index.get_level_values("unit")
    F = pd.get_dummies(year, prefix="y", drop_first=True, dtype=float).set_index(y.index)
    F["const"] = 1.0
    if not diff:
        F = F.join(pd.get_dummies(unit, prefix="u", drop_first=True, dtype=float).set_index(y.index))
    if trends:
        t = (year - start).values.astype(float)
        for u in units[1:]:
            F[f"t_{u}"] = t * (unit == u)
    return y, X.join(F), unit


def ols(y, X, unit):
    return sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(unit)[0]}, use_t=True)


def cumulative(r, lags=PRIMARY):
    c = np.zeros(len(r.params))
    for L in lags:
        c[r.params.index.get_loc(f"L{L}")] = 1.0
    t = r.t_test(c)
    return float(t.effect[0]), float(t.sd[0][0]), float(t.pvalue)


def randomization_p(tfr, rate, lags, start, end, diff, observed, seed):
    """Shuffle whole province marriage histories: national year effects and each series' shape stay intact."""
    rng = np.random.default_rng(seed)
    units = [c for c in rate.columns if c not in (NATIONAL, SEJONG)]
    draws = []
    for _ in range(RI_DRAWS):
        fake = rate.copy()
        fake[units] = rate[list(rng.permutation(units))].values
        y, X, _ = design(tfr, fake, lags, start, end, diff=diff)
        beta = np.linalg.lstsq(X.values, y.values, rcond=None)[0]
        draws.append(sum(beta[X.columns.get_loc(f"L{L}")] for L in PRIMARY))
    draws = np.array(draws)
    return float((np.abs(draws) >= abs(observed)).mean()), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def report(name, r, lags):
    rows = [dict(spec=name, term=f"L{L}", b=r.params[f"L{L}"], se=r.bse[f"L{L}"], p=r.pvalues[f"L{L}"]) for L in lags]
    b, se, p = cumulative(r)
    rows.append(dict(spec=name, term="L1+L2", b=b, se=se, p=p))
    return rows


def main():
    births, tfr, rate, monthly = load()
    assert (births.drop(columns=NATIONAL).sum(axis=1) - births[NATIONAL]).abs().max() == 0
    assert len([c for c in tfr.columns if c not in (NATIONAL, SEJONG)]) == 16

    rows = []
    lags = (0, 1, 2, 3)
    r1 = ols(*design(tfr, rate, lags, 2000, 2023))
    rows += report("R1 levels 2000~2023", r1, lags)
    lead_lags = (-2, -1, 0, 1, 2, 3)
    r2 = ols(*design(tfr, rate, lead_lags, 2000, 2021))
    rows += report("R2 leads 2000~2021", r2, lead_lags)
    lead_c = np.zeros(len(r2.params))
    lead_c[r2.params.index.get_loc("L-1")] = lead_c[r2.params.index.get_loc("L-2")] = 1.0
    t = r2.t_test(lead_c)
    rows.append(dict(spec="R2 leads 2000~2021", term="L-1+L-2", b=float(t.effect[0]), se=float(t.sd[0][0]), p=float(t.pvalue)))
    r3 = ols(*design(tfr, rate, lags, 2001, 2023, diff=True))
    rows += report("R3 differences 2001~2023", r3, lags)
    r4 = ols(*design(tfr, rate, lags, 2000, 2023, trends=True))
    rows += report("R4 province trends 2000~2023", r4, lags)
    T = pd.DataFrame(rows)
    T.to_csv(REP / "12_panel_coefficients.csv", index=False)
    print(T.round(3).to_string(index=False))

    print("\nrandomization inference on L1+L2")
    for name, r, diff, start, seed in (("R1", r1, False, 2000, 1), ("R3", r3, True, 2001, 3)):
        b = cumulative(r)[0]
        p, lo, hi = randomization_p(tfr, rate, lags, start, 2023, diff, b, seed)
        print(f"   {name}: observed {b:+.3f}, p={p:.3f}, shuffled 95% range {lo:+.3f}~{hi:+.3f}")

    loo = pd.Series({u: cumulative(ols(*design(tfr, rate, lags, 2000, 2023, drop=u)))[0] for u in tfr.columns if u not in (NATIONAL, SEJONG)})
    print(f"\nR1 leave-one-out L1+L2: {loo.min():+.3f} ({loo.idxmin()}) ~ {loo.max():+.3f} ({loo.idxmax()})")

    print("\nR5 counts with month windows, 2001~2023")
    b2 = births.copy()
    b2[CHUNGNAM] = b2[CHUNGNAM] + b2[SEJONG].fillna(0)
    m2 = monthly.drop(columns="국외").copy()
    m2[CHUNGNAM] = m2[CHUNGNAM] + m2[SEJONG].fillna(0)
    units = [c for c in b2.columns if c not in (NATIONAL, SEJONG)]
    yrs = list(range(2001, 2024))
    out = []
    for lo, hi in ((12, 35), (9, 32), (14, 37)):
        w = sum(m2.shift(k) for k in range(lo, hi + 1))
        annual = w.groupby(w.index.year).sum(min_count=12)
        abroad = sum(monthly["국외"].shift(k) for k in range(lo, hi + 1))
        assert abs(annual.loc[2010, units].sum() + abroad["2010"].sum() - annual.loc[2010, NATIONAL]) < 1  # provinces + abroad = national
        y = np.log(b2.loc[yrs, units]).stack()
        y.index.names = ["year", "unit"]
        x = np.log(annual.loc[yrs, units]).stack().reindex(y.index).rename("marr")
        unit = y.index.get_level_values("unit")
        F = pd.get_dummies(y.index.get_level_values("year"), prefix="y", drop_first=True, dtype=float).set_index(y.index)
        F = F.join(pd.get_dummies(unit, prefix="u", drop_first=True, dtype=float).set_index(y.index))
        F["const"] = 1.0
        X = F.join(x)
        r = ols(y, X, unit)
        within = 1 - r.ssr / sm.OLS(y, F).fit().ssr
        out.append(dict(window=f"{lo}~{hi}", b=r.params["marr"], se=r.bse["marr"], p=r.pvalues["marr"], aic=r.aic, within_r2=within))
    R5 = pd.DataFrame(out)
    R5.to_csv(REP / "12_panel_counts.csv", index=False)
    print(R5.round(3).to_string(index=False))

    print("\nR6 2023->2025 province differences from coefficients fitted through 2023 (L1~L3)")
    fit_lags = (1, 2, 3)
    r6 = ols(*design(tfr, rate, fit_lags, 2000, 2023))
    print("   coefficients", {f"L{L}": round(r6.params[f"L{L}"], 3) for L in fit_lags})
    units = [c for c in tfr.columns if c not in (NATIONAL, SEJONG)]
    g = np.log(rate[units])
    predicted = sum(r6.params[f"L{L}"] * (g.loc[2025 - L] - g.loc[2023 - L]) for L in fit_lags)
    actual = np.log(tfr.loc[2025, units] / tfr.loc[2023, units])
    C = pd.DataFrame({"predicted": predicted - predicted.mean(), "actual": actual - actual.mean(), "tfr_2023": tfr.loc[2023, units], "tfr_2025": tfr.loc[2025, units], "marr_change_22_24": g.loc[2024] - g.loc[2022]})
    C.to_csv(REP / "12_rebound_provinces.csv")
    print((C * [100, 100, 1, 1, 100]).round(2).sort_values("actual").to_string())
    fit = sm.OLS(C.actual, sm.add_constant(C.predicted)).fit()
    rmse = lambda e: float(np.sqrt((e**2).mean())) * 100
    print(f"   corr {C.predicted.corr(C.actual):.3f}, slope {fit.params['predicted']:.2f} (se {fit.bse['predicted']:.2f}, p {fit.pvalues['predicted']:.3f}), RMSE {rmse(C.actual - C.predicted):.2f} vs no-difference {rmse(C.actual):.2f}, n={len(C)}")
    print(f"   rank corr {C.predicted.corr(C.actual, method='spearman'):.3f}; national log TFR change {np.log(tfr.loc[2025, NATIONAL] / tfr.loc[2023, NATIONAL]) * 100:.2f}%")

    # added after seeing R6, report only: the largest predicted miss, and a rough national reading of the elasticity
    rest = C.drop(index="대전광역시")
    rest = rest - rest.mean()
    print(f"post hoc, without Daejeon: corr {rest.predicted.corr(rest.actual):.3f}, RMSE {rmse(rest.actual - rest.predicted):.2f} vs no-difference {rmse(rest.actual):.2f}, n={len(rest)}")
    g_nat = np.log(rate[NATIONAL])
    implied = sum(r6.params[f"L{L}"] * (g_nat[2025 - L] - g_nat[2023 - L]) for L in fit_lags)
    total = np.log(tfr.loc[2025, NATIONAL] / tfr.loc[2023, NATIONAL])
    print(f"post hoc, national 2023->2025 at panel coefficients: {implied * 100:+.2f}% of {total * 100:+.2f}% ({implied / total:.0%}); L1 alone {r6.params['L1'] * (g_nat[2024] - g_nat[2022]) * 100:+.2f}%")


if __name__ == "__main__":
    main()
