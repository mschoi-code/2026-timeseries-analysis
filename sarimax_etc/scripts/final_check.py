"""Attribute the ~7% rebound MAPE: exogenous variables vs d=0 itself, window placebo, lag profile, calendar."""
import sys
sys.path.append(sys.argv[2])  # appended last so stdlib and site-packages win
from spec_search import *  # noqa: E402,F403
import statsmodels.api as sm  # noqa: E402

SPEC = dict(order=(1, 0, 0), sorder=(0, 1, 1, 12), trend="c")   # the grid spec closest to 7.1 (7.09)
BASE_ = dict(order=(0, 1, 1), sorder=(0, 1, 1, 12), trend=None)
def score(X, s):
    fit, f, ac = fc(X, "2023-12", 31, **s)
    nm = [mape(*fc(X, f"{yr}-12", 12, **s)[1:]) for yr in range(2015, 2023)]
    return mape(f, ac), float(np.mean(nm)), fit

act = np.exp(y)
sn = pd.Series(act["2023"].values, index=range(1, 13))
fix = act["2024-01":"2026-07"]
print("T0 fixed-origin seasonal naive (repeat 2023): rebound MAPE %.1f" % mape(pd.Series(sn.reindex(fix.index.month).values, index=fix.index), fix))
nwf = np.log(nw)
wf = np.log(wl)
rows = {"d=0, no X": None, "d=0, newlywed": nwf.to_frame("a"), "d=0, women": wf.to_frame("a"),
        "d=0, newlywed+women": pd.DataFrame({"a": nwf, "b": wf})}
for k, X in rows.items():
    r, n, fit = score(X, SPEC)
    co = "" if X is None else " ".join(f"{c}={fit.params[c]:+.2f}(se {fit.bse[c]:.2f})" for c in X.columns)
    print(f"T1 (1,0,0)(0,1,1)+c [{k}]: rebound {r:.1f} | normal {n:.2f} | {co}")
r, n, _ = score(None, BASE_)
print(f"T1 (0,1,1)(0,1,1) [SARIMA]: rebound {r:.1f} | normal {n:.2f}")

for lo, hi in [(12, 35), (24, 47), (36, 59), (48, 71), (60, 83)]:
    x = np.log(sum(full.marr.shift(k) for k in range(lo, hi + 1)) / 2)
    X = pd.DataFrame({"a": x, "b": wf})
    r, n, fit = score(X, SPEC)
    print(f"T2 window {lo}~{hi}m earlier, d=0 + women: rebound {r:.1f} | normal {n:.2f} | b_marr {fit.params['a']:+.2f} (se {fit.bse['a']:.2f})")

# which individual marriage lags carry signal, in the differenced space the Granger slide relies on
lb, lmr = np.log(m.births).diff().diff(12), np.log(m.marr).diff().diff(12)
D = pd.DataFrame({"y": lb, **{f"y{k}": lb.shift(k) for k in (1, 2, 12, 13)}, **{f"m{k}": lmr.shift(k) for k in range(0, 37)}})["2006":"2023"].dropna()
ols = sm.OLS(D.y, sm.add_constant(D.drop(columns="y"))).fit()
ts = ols.tvalues.filter(regex="^m")
print("T3 lags with |t|>2 (2006-2023):", {k: round(v, 2) for k, v in ts[ts.abs() > 2].items()}, "| expected false positives ~", round(37 * 0.05, 1))
win = [f"m{k}" for k in range(12, 36)]
print("T3b joint F lags 12~35 p=%.4f | lags 0~11 p=%.4f" % (float(ols.f_test(" = ".join(win) + " = 0").pvalue), float(ols.f_test(" = ".join(f"m{k}" for k in range(0, 12)) + " = 0").pvalue)))

# weekday-count calendar effect (holidays not included)
idx = m.index
wd = pd.Series([np.busday_count(d.date(), (d + pd.DateOffset(months=1)).date()) for d in idx], index=idx, dtype=float)
dw = np.log(wd).diff().diff(12)
print("T4 dd corr with weekday count 2006-2023: births %+.3f | marriages %+.3f" % (lb["2006":"2023"].corr(dw["2006":"2023"]), lmr["2006":"2023"].corr(dw["2006":"2023"])))
print("T4b dd births~marr(t) corr %+.3f | partial on weekdays %+.3f" % (lb["2006":"2023"].corr(lmr["2006":"2023"]),
      sm.OLS(lb["2006":"2023"], sm.add_constant(dw["2006":"2023"])).fit().resid.corr(sm.OLS(lmr["2006":"2023"], sm.add_constant(dw["2006":"2023"])).fit().resid)))
f0 = SARIMAX(y[:"2023-12"], order=(0, 1, 1), seasonal_order=(0, 1, 1, 12)).fit(disp=False)
f1 = SARIMAX(y[:"2023-12"], exog=np.log(wd).loc[y[:"2023-12"].index], order=(0, 1, 1), seasonal_order=(0, 1, 1, 12)).fit(disp=False)
print("T4c SARIMA AIC %.1f -> with log weekdays %.1f | b %+.3f se %.3f" % (f0.aic, f1.aic, f1.params.iloc[0], f1.bse.iloc[0]))
