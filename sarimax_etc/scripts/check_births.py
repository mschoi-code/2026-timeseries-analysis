"""One-off check of the team's raw birth/marriage/population files against the proposal's numbers."""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import grangercausalitytests

warnings.filterwarnings("ignore")
pd.set_option("display.width", 160)
raw = Path(sys.argv[1])

# --- A. monthly births / marriages
m = pd.read_csv(raw / "월간_인구동향_출생_혼인.csv", skiprows=2, header=None,
                names=["t", "births", "marr"], dtype={"t": str})
m["prelim"] = m.t.str.contains("p)", regex=False)
m["date"] = pd.to_datetime(m.t.str.slice(0, 7), format="%Y.%m")
m = m.set_index("date")
idx = pd.date_range(m.index.min(), m.index.max(), freq="MS")
assert m.index.equals(idx), "gaps or disorder in months"
assert m[["births", "marr"]].notna().all().all() and (m[["births", "marr"]] > 0).all().all()
print("A1 rows", len(m), m.index.min().date(), "~", m.index.max().date(), "| dtypes", m.births.dtype, m.marr.dtype)
print("A2 prelim months:", m.index[m.prelim].strftime("%Y-%m").tolist())

ann = m[["births", "marr"]].groupby(m.index.year).sum()
ann["b_yoy%"] = ann.births.pct_change() * 100
ann["m_yoy%"] = ann.marr.pct_change() * 100
print("A3 annual\n", ann.loc[[1992, 2005, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]].round(1).to_string())
print("A3b years with births increase 2010-2025:", ann.index[(ann["b_yoy%"] > 0) & (ann.index >= 2010)].tolist())
jj = m[m.index.month <= 7].groupby(lambda d: d.year)[["births", "marr"]].sum()
print("A4 Jan-Jul\n", jj.loc[2023:].assign(b_yoy=lambda d: d.births.pct_change() * 100, m_yoy=lambda d: d.marr.pct_change() * 100).round(1).to_string())
ma = m.births.rolling(12).mean()
print("A5 12m avg low", ma["2005":].idxmin().date(), round(ma["2005":].min()), "| Jan17", m.births["2017-01"].iloc[0], "| 12m avg at 2026-07", round(ma.iloc[-1]),
      "| min monthly 2017+", m.births["2017":].idxmin().date(), m.births["2017":].min())

ratio = m.births / m.births.rolling(12, center=True).mean().rolling(2).mean().shift(-1)
for lab, a, b in [("1995-99", "1995", "1999"), ("2005-14", "2005", "2014"), ("2015-19", "2015", "2019"), ("2020-25", "2020", "2025")]:
    s = ratio[a:b].groupby(ratio[a:b].index.month).mean()
    print(f"A6 seasonal {lab}: Jan {100*(s[1]-1):+.1f}% Sep {100*(s[9]-1):+.1f}% Dec {100*(s[12]-1):+.1f}%")
amp = m.births.groupby(m.index.year).agg(lambda s: s.max() - s.min())
rel = amp / m.births.groupby(m.index.year).mean() * 100
print("A7 amplitude abs:", {y: int(amp[y]) for y in (2005, 2010, 2015, 2020, 2023, 2024, 2025)}, "\n   rel% range 2005-25:", round(rel.loc[2005:2025].min(), 1), round(rel.loc[2005:2025].max(), 1))
print("A7b max-min by year:", {y: (int(amp[y]), round(rel[y], 1)) for y in range(2005, 2026)})
print("A5b 12m avg 2017-01:", round(ma["2017-01-01"]), "| first-year max-min peak", amp.loc[2005:].idxmax(), int(amp.loc[2005:].max()))

for c in ("births", "marr"):
    dd = np.log(m[c]).diff().diff(12)["2005":]
    z = (dd - dd.median()) / (1.4826 * (dd - dd.median()).abs().median())
    top = z.abs().sort_values(ascending=False).head(6).index.sort_values()
    print(f"A8 outliers {c} (robust z of dlog-d12):", {d.strftime('%Y-%m'): round(z[d], 1) for d in top}, "| |z|>3:", int((z.abs() > 3).sum()), "of", len(z))

# --- B. newlywed couples (marriages 12~35 months earlier / 2)
full = m.reindex(pd.date_range(m.index.min(), m.index.max() + pd.DateOffset(months=12), freq="MS"))
nw = sum(full.marr.shift(k) for k in range(12, 36)) / 2
assert nw["2005":].notna().all()
assert abs(nw["2026-07-01"] - m.marr["2023-08":"2025-07"].sum() / 2) < 1e-6
print("B1 newlywed span", nw.first_valid_index().date(), "~", nw.last_valid_index().date(), "| low", nw["2005":].idxmin().date(), round(nw["2005":].min()), "| 2026-07", round(nw["2026-07-01"]), "| 2027-07", round(nw["2027-07-01"]))
b12 = m.births.rolling(12).sum()
r = (b12 / nw).dropna()
print("B2 births(12m)/newlywed:", {y: round(r[f"{y}-12-01"], 3) for y in (2010, 2015, 2019, 2020, 2021, 2022, 2023, 2024, 2025)}, "2026-07", round(r.iloc[-1], 3))

# --- C. population projection (women)
p = pd.read_csv(raw / "성_및_연령별_추계인구_1세별_5세별_전국.csv", header=None, dtype=str)
assert set(p.iloc[0, 1:]) .__len__() == 1 and set(p.iloc[1, 1:]) == {"여자"}
print("C0 scenario:", p.iloc[0, 1], "| rows", p.iloc[3:, 0].tolist()[:3], "...", p.iloc[-1, 0], "n", len(p) - 3)
years = p.iloc[2, 1:].astype(int).values
pop = p.iloc[3:].set_index(0)
pop.columns = years
pop = pop.replace("-", np.nan).astype(float)
assert list(years) == list(range(years[0], years[-1] + 1))
print("C1 years", years[0], "~", years[-1], "| NaN years per group:", pop.isna().sum(axis=1).loc[lambda s: s > 0].to_dict())
print("C2a labels:", pop.index.tolist())
five = [i for i in pop.index if i != "계" and i != "80세 이상"]
gap = (pop.loc[five].sum() - pop.loc["계"]).loc[2000:]
print("C2 sum(5yr)-total max abs 2000+:", gap.abs().max())
w = pop.loc["30 - 34세"]
print("C3 women30-34:", {y: int(w[y]) for y in (2005, 2010, 2015, 2021, 2022, 2023, 2024, 2025, 2026, 2027, 2031, 2035)}, "| peak", w.loc[2005:].idxmax(), "| 2035 vs 2026", round((w[2035] / w[2026] - 1) * 100, 1), "%")
print("C4 yoy% 2018-2028:", {y: round((w[y] / w[y - 1] - 1) * 100, 1) for y in range(2018, 2029)})

# mid-year anchored linear interpolation to monthly (one reasonable choice; the proposal leaves it open)
wm = pd.Series(w.values, index=pd.to_datetime([f"{y}-07-01" for y in years]))
wm = wm.reindex(pd.date_range(wm.index[0], wm.index[-1], freq="MS")).interpolate("linear")

# --- D. collinearity of the two exogenous variables
X = pd.DataFrame({"nw": np.log(nw), "w": np.log(wm), "y": np.log(m.births)})
for a, b in [("2005", "2015"), ("2005", "2023")]:
    c = X[a:b][["nw", "w"]].corr().iloc[0, 1]
    print(f"D1 level corr nw~w {a}-{b}: {c:.3f} VIF {1/(1-c**2):.1f}")
dd = X.diff().diff(12)
for a, b in [("2006", "2015"), ("2006", "2023")]:
    print(f"D2 dd corr {a}-{b}: nw~w {dd[a:b].nw.corr(dd[a:b].w):+.3f} | y~nw {dd[a:b].y.corr(dd[a:b].nw):+.3f} | y~w {dd[a:b].y.corr(dd[a:b].w):+.3f} | sd y {dd[a:b].y.std():.4f} nw {dd[a:b].nw.std():.4f} w {dd[a:b].w.std():.5f}")
print("D3 distinct dd-w values 2006-2023 (rounded 1e-6):", dd["2006":"2023"].w.round(6).nunique(), "of", len(dd["2006":"2023"]))

# --- E. reproduce the proposal's EDA(2): SARIMA trained 2005~2023
y = np.log(m.births["2005":])
fit = SARIMAX(y[:"2023-12"], order=(0, 1, 1), seasonal_order=(0, 1, 1, 12)).fit(disp=False)
fc = fit.get_forecast(31)
ci, mean = fc.conf_int(alpha=0.05), fc.predicted_mean
act = y["2024-01":"2026-07"]
assert len(act) == 31 and act.index.equals(mean.index)
above = act > ci.iloc[:, 1]
below = act < ci.iloc[:, 0]
print("E1 params", fit.params.round(3).to_dict(), "AIC", round(fit.aic, 1))
print("E2 outside 95% PI:", int(above.sum() + below.sum()), "of 31 | above", int(above.sum()), "below", int(below.sum()), "| first above", above[above].index[0].date() if above.any() else None)
print("E3 2026 months above:", int(above["2026"].sum()), "of", len(above["2026"]))
ape = (np.exp(mean) / np.exp(act) - 1) * 100
print("E4 mean forecast error % by year:", ape.groupby(ape.index.year).mean().round(1).to_dict())

# --- F. Granger marriage <-> births over lags 3~24 under several transforms and windows
lm = np.log(m[["births", "marr"]])
tf = {"log level": lm, "d12": lm.diff(12), "d1d12": lm.diff().diff(12)}
def pvals(df, cause, effect):
    r = grangercausalitytests(df[[effect, cause]].dropna(), maxlag=24)
    return pd.Series({k: r[k][0]["ssr_ftest"][1] for k in range(3, 25)})
for name, d in tf.items():
    for a, b in [("2005", "2026"), ("2005", "2015")]:
        sub = d[a:b]
        f = pvals(sub, "marr", "births"); g = pvals(sub, "births", "marr")
        print(f"F {name:9s} {a}-{b}: marr->births p<.01 {int((f<.01).sum())}/22 (max p {f.max():.3f}) | births->marr p<.01 {int((g<.01).sum())}/22, p<.05 {int((g<.05).sum())}/22 (min p {g.min():.3f})")

# --- G. press cross-check for 2026-08 (published 2026-10-28): births 23,703, marriages 19,218
print("G1 file Jan-Jul 2026 + press Aug births:", int(m.births["2026-01":"2026-07"].sum() + 23703), "(press cum 193,783)")
base = m.marr["2025-01":"2025-08"].sum()
print("G2 Aug 2025 births/marr in file:", int(m.births["2025-08-01"]), int(m.marr["2025-08-01"]),
      "| implied births yoy %.1f (press 13.0), marr yoy %.1f (press -1.2), marr cum yoy %.1f (press 5.2)" % (
      (23703 / m.births["2025-08-01"] - 1) * 100, (19218 / m.marr["2025-08-01"] - 1) * 100,
      ((m.marr["2026-01":"2026-07"].sum() + 19218) / base - 1) * 100))

# --- H. seasonal amplitude after detrending (slide: 12,500 -> 6,200 and 22~29%)
for a, b in [("1990", "1999"), ("2000", "2004"), ("2005", "2009"), ("2010", "2014"), ("2015", "2019"), ("2020", "2023"), ("2021", "2025")]:
    s_ = ratio[a:b].groupby(ratio[a:b].index.month).mean()
    lvl = m.births[a:b].mean()
    print(f"H {a}-{b}: rel {100*(s_.max()-s_.min()):.1f}% abs {(s_.max()-s_.min())*lvl:,.0f} (level {lvl:,.0f})")

# --- I. reproduce the headline: SARIMA vs SARIMAX with newlywed + women 30~34, same order, trained through 2023
def run(y, X, end, h, order=(0, 1, 1), sorder=(0, 1, 1, 12), trend=None):
    tr = y[:end]
    te_idx = pd.date_range(tr.index[-1] + pd.DateOffset(months=1), periods=h, freq="MS")
    te_idx = te_idx[te_idx <= y.index[-1]]
    fit = SARIMAX(tr, exog=None if X is None else X.loc[tr.index], order=order, seasonal_order=sorder, trend=trend).fit(disp=False, maxiter=200)
    f = fit.get_forecast(len(te_idx), exog=None if X is None else X.loc[te_idx]).predicted_mean
    return fit, np.exp(f), np.exp(y.loc[te_idx])
Xall = pd.DataFrame({"lnw": np.log(nw), "lw": np.log(wm), "lm": np.log(m.marr)})
def mape(f, a): return float((abs(f - a) / a).mean() * 100)
variants = {"SARIMA": None, "marriage (t)": ["lm"], "newlywed": ["lnw"], "women30-34": ["lw"], "newlywed+women": ["lnw", "lw"]}
for name, cols in variants.items():
    fit, f, a = run(y, None if cols is None else Xall[cols], "2023-12", 31)
    co = "" if cols is None else " ".join(f"{c}={fit.params[c]:+.3f}(se {fit.bse[c]:.3f}, p {fit.pvalues[c]:.3f})" for c in cols)
    print(f"I1 rebound MAPE 2024.01-2026.07 [{name}]: {mape(f, a):.1f}% | by year {((f/a-1)*100).groupby(a.index.year).mean().round(1).to_dict()} | {co}")

# coefficient stability across expanding windows, and the normal-period test the proposal will select on
rows = []
for yr in range(2015, 2023):
    end = f"{yr}-12"
    s0 = run(y, None, end, 12); s1 = run(y, Xall[["lnw", "lw"]], end, 12)
    rows.append({"target": yr + 1, "SARIMA": mape(s0[1], s0[2]), "SARIMAX": mape(s1[1], s1[2]),
                 "b_nw": s1[0].params["lnw"], "se_nw": s1[0].bse["lnw"], "b_w": s1[0].params["lw"], "se_w": s1[0].bse["lw"]})
R = pd.DataFrame(rows).set_index("target")
print("I2 normal period, 8 origins (Dec of previous year):\n", R.round(2).to_string())
print("I3 mean MAPE: SARIMA %.2f | SARIMAX %.2f | SARIMAX wins %d/8" % (R.SARIMA.mean(), R.SARIMAX.mean(), (R.SARIMAX < R.SARIMA).sum()))
