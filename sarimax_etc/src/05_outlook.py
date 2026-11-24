"""Step 5, RQ3 outlook to 2031 by marriage scenario, refit through 2026-07, plus a long-horizon backtest.

Scenarios were declared in notes/decisions.md before this ran.
"""
import json
import sys
import warnings
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))  # appended last: scripts run with python -I
from common import BLUE, FIG, GREEN, INK, MUTED, ORANGE, REBOUND_END, REP, fit, load_features, load_monthly, newlywed, plt, weekdays, women_30_34  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

warnings.filterwarnings("ignore")
pd.set_option("display.width", 170)
SCENARIOS = {"High": 0.05, "Hold": 0.0, "Low": -0.05}
EXOG = ["weekdays", "newlywed", "women"]

m = load_monthly()
y, _ = load_features()
y = y[:REBOUND_END]
assert m.index[-1] == y.index[-1], "outlook starts right after the last observation"
level = np.exp(y)
cands = {(c["track"], c["model"]): c["spec"] for c in json.loads((REP / "01_candidates.json").read_text())}
lock = json.loads((REP / "02_locked.json").read_text())
MODELS = {"d=1 main": lock["best_each_d"]["1"], "d=0 alternative": lock["best_each_d"]["0"]}
assert lock["main"]["model"] == MODELS["d=1 main"]["model"] and all(v["track"] == "weekdays" for v in MODELS.values())


def spec_of(v):
    s = cands[(v["track"], v["model"])]
    return dict(order=tuple(s["order"]), seasonal_order=tuple(s["seasonal_order"]), trend=s["trend"])


def extend(marr, growth, end):
    """Each future month = same month a year earlier x (1 + growth)."""
    out = marr.copy()
    for t in pd.date_range(marr.index[-1] + pd.DateOffset(months=1), end, freq="MS"):
        out[t] = out[t - pd.DateOffset(months=12)] * (1 + growth)
    return out


def regressors(marr, end):
    """Regressors through end; marriages must already reach 12 months before end."""
    X = pd.DataFrame({"newlywed": np.log(newlywed(marr)), "women": np.log(women_30_34())}).dropna()
    X["weekdays"] = np.log(weekdays(X.index))
    assert X.index[-1] >= end
    return X[:end]


# --- outlook
END = pd.Timestamp("2031-12-01")
future = pd.date_range(y.index[-1] + pd.DateOffset(months=1), END, freq="MS")
X_by = {s: regressors(extend(m.marr, g, pd.Timestamp("2030-12-01")), END) for s, g in SCENARIOS.items()}
known_until = "2027-07"  # marriages through 2026-07 fix newlywed for 12 more months
assert X_by["High"][:known_until].equals(X_by["Low"][:known_until]), "scenarios must agree while newlywed is already known"
assert not X_by["High"].newlywed["2027-08":].equals(X_by["Low"].newlywed["2027-08":])

monthly, annual, params = [], [], []
base_2026 = level["2026"].sum()
for mname, v in MODELS.items():
    spec = spec_of(v)
    for sname, X in X_by.items():
        res = fit(y, X[EXOG], spec)
        assert res.mle_retvals["converged"]
        fc = res.get_forecast(len(future), exog=X.loc[future, EXOG])
        point = np.exp(fc.predicted_mean)
        ci = fc.conf_int(alpha=0.05)
        monthly.append(pd.DataFrame(dict(model=mname, scenario=sname, target=future, forecast=point.values, lo=np.exp(ci.iloc[:, 0]).values, hi=np.exp(ci.iloc[:, 1]).values)))
        sim = np.exp(res.simulate(len(future), anchor="end", repetitions=2000, exog=X.loc[future, EXOG], rng=np.random.default_rng(20261106)))
        sim.index = future
        sums = sim.groupby(sim.index.year).sum()
        sums.loc[2026] += base_2026
        pt = point.groupby(point.index.year).sum()
        pt.loc[2026] += base_2026
        annual.append(pd.DataFrame(dict(model=mname, scenario=sname, year=pt.index, births=pt.values, lo=sums.quantile(0.025, axis=1).values, hi=sums.quantile(0.975, axis=1).values)))
        params.append(dict(model=mname, scenario=sname, b_newlywed=res.params["newlywed"], se_newlywed=res.bse["newlywed"], b_women=res.params["women"], se_women=res.bse["women"], b_weekdays=res.params["weekdays"]))
        if sname == "Hold":
            print(f"O0 {mname}: forecast 2026-08 {point.iloc[0]:,.0f} vs press-reported 23,703 ({(point.iloc[0] / 23703 - 1) * 100:+.1f}%)")
A = pd.concat(annual, ignore_index=True)
pd.concat(monthly, ignore_index=True).to_csv(REP / "05_outlook_monthly.csv", index=False)
A.to_csv(REP / "05_outlook_annual.csv", index=False)
print("O1 coefficients on 2005~2026-07\n", pd.DataFrame(params).round(3).to_string(index=False))
print("O2 annual births, thousands\n", (A.pivot_table(index="year", columns=["model", "scenario"], values="births", sort=False) / 1000).round(1).to_string())
print("O3 95% simulation interval under Hold, thousands\n", (A[A.scenario == "Hold"].pivot_table(index="year", columns="model", values=["lo", "hi"], sort=False) / 1000).round(0).to_string())
print("O4 peak year by model and scenario:", {f"{k[0]}|{k[1]}": int(g.loc[g.births.idxmax(), "year"]) for k, g in A.groupby(["model", "scenario"], sort=False)})

# --- how far can a 5-year outlook be trusted? December origins with marriages held after the origin
rows = []
actual_year = m.births.groupby(m.index.year).sum()
for o_year in range(2015, 2021):
    origin = pd.Timestamp(f"{o_year}-12-01")
    tr, marr = y[:origin], m.marr[:origin]
    last12 = actual_year[o_year]
    end = origin + pd.DateOffset(months=60)
    X = regressors(extend(marr, 0.0, end - pd.DateOffset(months=12)), end)
    idx = pd.date_range(origin + pd.DateOffset(months=1), end, freq="MS")
    for mname, v in MODELS.items():
        for kind, cols in (("SARIMAX", EXOG), ("SARIMA", ["weekdays"])):
            r = fit(tr, X[cols], spec_of(v))
            f = np.exp(r.get_forecast(60, exog=X.loc[idx, cols]).predicted_mean)
            ann = f.groupby(f.index.year).sum()
            rows += [dict(name=f"{mname} {kind}", origin=o_year, ahead=yr - o_year, error=(ann[yr] / actual_year[yr] - 1) * 100) for yr in ann.index]
    rows += [dict(name="last year repeated", origin=o_year, ahead=yr - o_year, error=(last12 / actual_year[yr] - 1) * 100) for yr in range(o_year + 1, o_year + 6)]
B = pd.DataFrame(rows)
B.to_csv(REP / "05_backtest_annual.csv", index=False)
print("O5 backtest, mean absolute annual error % by years ahead, 6 December origins\n", B.assign(abs_error=B.error.abs()).pivot_table(index="name", columns="ahead", values="abs_error", sort=False).round(1).to_string())
print("O6 signed error at 5 years ahead by origin\n", B[B.ahead == 5].pivot_table(index="name", columns="origin", values="error", sort=False).round(1).to_string())

# --- slide figure: annual births by scenario, one panel per model on one shared scale
hist = actual_year.loc[2015:2025] / 1000
colors = {"High": GREEN, "Hold": BLUE, "Low": ORANGE}
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
for ax, (mname, g) in zip(axes, A.groupby("model", sort=False)):
    ax.plot(hist.index, hist.values, color=INK, label="Actual")
    hold = g[g.scenario == "Hold"]
    ax.fill_between(hold.year, hold.lo / 1000, hold.hi / 1000, color=BLUE, alpha=0.12, lw=0, label="Hold 95% interval")
    for sname, s in g.groupby("scenario", sort=False):
        ax.plot([2025] + list(s.year), [hist[2025]] + list(s.births / 1000), color=colors[sname], label=sname)
        ax.annotate(sname, (2031, s.births.iloc[-1] / 1000), xytext=(5, {"High": 7, "Hold": 0, "Low": -7}[sname]), textcoords="offset points", fontsize=9, va="center")
    ax.set_title(mname, loc="left", fontsize=11)
    ax.set_xticks(range(2015, 2032, 4))
    ax.set_xlabel("Year (2026 includes 5 forecast months)")
axes[0].set_ylabel("Annual births (thousands)")
axes[0].set_ylim(0, None)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.07))
fig.suptitle("Birth outlook to 2031 by marriage scenario", x=0.01, ha="left", fontsize=12)
fig.tight_layout()
fig.savefig(FIG / "05_outlook.png")
