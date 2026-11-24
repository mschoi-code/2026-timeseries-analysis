"""How much of the birth rebound do marriages explain? Read-only decomposition; changes no model choice."""
import json
import sys
import warnings
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))  # appended last: scripts run with python -I
from common import DATA, REP, TRAIN_END, fit, load_features, load_monthly, mape, newlywed, weekdays  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

warnings.filterwarnings("ignore")
pd.set_option("display.width", 170)
m = load_monthly()
p = pd.read_csv(DATA / "성_및_연령별_추계인구_1세별_5세별_전국.csv", header=None, dtype=str)
years = p.iloc[2, 1:].astype(int).values
P = p.iloc[3:].set_index(0)


def monthly(group):
    s = pd.Series(P.loc[group].astype(float).values, index=pd.to_datetime([f"{y}-07-01" for y in years]))
    return s.reindex(pd.date_range(s.index[0], s.index[-1], freq="MS")).interpolate("linear")


w2529, w3034, w3539 = monthly("25 - 29세"), monthly("30 - 34세"), monthly("35 - 39세")
w2534 = w2529 + w3034
nw = newlywed(m.marr)
lag_pop = sum(w2534.shift(k) for k in range(12, 36)) / 24  # women 25~34 when those couples married

periods = {str(y): (f"{y}-01", f"{y}-12") for y in (2015, 2019, 2021, 2022, 2023, 2024, 2025)}
periods["L12 (25.08~26.07)"] = ("2025-08", "2026-07")
rows = []
for k, (a, b) in periods.items():
    rows.append(dict(period=k, births=m.births[a:b].sum(), marriages=m.marr[a:b].sum(), newlywed=nw[a:b].mean(), w2534=w2534[a:b].mean(), w3034=w3034[a:b].mean(),
                     lag_w2534=lag_pop[a:b].mean()))
T = pd.DataFrame(rows).set_index("period")
T["births_per_newlywed"] = T.births / T.newlywed
T["marr_per_1k_w2534"] = T.marriages / T.w2534 * 1000
T["births_per_1k_w3034"] = T.births / T.w3034 * 1000
T["nw_per_1k_lag_w2534"] = T.newlywed / T.lag_w2534 * 1000
print("D1 levels\n", T.round({"births": 0, "marriages": 0, "newlywed": 0, "w2534": 0, "w3034": 0, "lag_w2534": 0, "births_per_newlywed": 3, "marr_per_1k_w2534": 1, "births_per_1k_w3034": 1, "nw_per_1k_lag_w2534": 1}).T.to_string())

b, e = "2023", "L12 (25.08~26.07)"
d = lambda c: np.log(T.loc[e, c] / T.loc[b, c]) * 100
print(f"\nD2 log change 2023 -> L12, %: births {d('births'):.1f} = newlywed {d('newlywed'):.1f} + births per newlywed {d('births_per_newlywed'):.1f}")
print(f"   share of the birth rise: newlywed {d('newlywed') / d('births') * 100:.0f}%, births per newlywed {d('births_per_newlywed') / d('births') * 100:.0f}%")
print(f"   births {d('births'):.1f} = women 30~34 {d('w3034'):.1f} + births per woman 30~34 {d('births_per_1k_w3034'):.1f}")
print(f"   newlywed {d('newlywed'):.1f} = lagged women 25~34 {d('lag_w2534'):.1f} + lagged marriage rate {d('nw_per_1k_lag_w2534'):.1f}")
print(f"   marriages 2023->L12 {d('marriages'):.1f} = women 25~34 {d('w2534'):.1f} + crude rate {d('marr_per_1k_w2534'):.1f}")

# turning points of trailing 12-month sums
roll = {"marriages": m.marr.rolling(12).sum()["2015":], "births": m.births.rolling(12).sum()["2015":], "newlywed": nw["2015":"2026-07"]}
print("\nD3 troughs:", {k: v.idxmin().strftime("%Y-%m") for k, v in roll.items()})
print("   window-free lag: births trough minus marriage trough, months:", (roll["births"].idxmin().year - roll["marriages"].idxmin().year) * 12 + roll["births"].idxmin().month - roll["marriages"].idxmin().month)

# model attribution with coefficients frozen at 2023-12, and with the refit through 2026-07
y, X = load_features()
span = lambda s, a, b: s[a:b].mean()
dx = {c: (span(X[c], "2025-08", "2026-07") - span(X[c], "2023-01", "2023-12")) * 100 for c in ("newlywed", "women")}
dy = (span(y, "2025-08", "2026-07") - span(y, "2023-01", "2023-12")) * 100
C = pd.read_csv(REP / "03_coefficients.csv").set_index("name")
print(f"\nD4 mean log change 2023 -> L12: births {dy:.1f}, newlywed {dx['newlywed']:.1f}, women {dx['women']:.1f}")
LOCK = json.loads((REP / "02_locked.json").read_text())
for name, label_ in ((f"{LOCK['best_each_d']['0']['model']}|weekdays|SARIMAX", "d=0"), (f"{LOCK['best_each_d']['1']['model']}|weekdays|SARIMAX", "d=1")):
    c = C.loc[name]
    a, b_ = c.b_newlywed * dx["newlywed"], c.b_women * dx["women"]
    print(f"   {label_} frozen at 2023-12: newlywed {a:.1f} ({a / dy * 100:.0f}%), women {b_:.1f} ({b_ / dy * 100:.0f}%), left {dy - a - b_:.1f} ({(dy - a - b_) / dy * 100:.0f}%)")

# which variable carries the d=0 gain? same frozen 20-origin metric as the proposal
level = np.exp(y[:"2026-07"])
y_ = y[:"2026-07"]
ORIGINS = pd.date_range(TRAIN_END, "2025-07-01", freq="MS")
spec = dict(order=(1, 0, 0), seasonal_order=(0, 1, 1, 12), trend="c")
out = {}
for name, cols in {"weekdays only": ["weekdays"], "+ newlywed": ["weekdays", "newlywed"], "+ women": ["weekdays", "women"], "+ both": ["weekdays", "newlywed", "women"]}.items():
    ex = X[cols]
    res = fit(y_[:TRAIN_END], ex, spec)
    scores = []
    for o in ORIGINS:
        r = res.apply(y_[:o], exog=ex.loc[y_[:o].index])
        idx = pd.date_range(o + pd.DateOffset(months=1), periods=12, freq="MS")
        scores.append(mape(np.exp(r.get_forecast(12, exog=ex.loc[idx]).predicted_mean), level[idx]))
    coef = {c: f"{res.params[c]:+.2f} ({res.bse[c]:.2f})" for c in cols if c != "weekdays"}
    out[name] = float(np.mean(scores))
    print(f"D5 d=0 {name:14s} rebound MAPE {out[name]:.2f}  {coef}")
S = pd.read_csv(REP / "03_rebound_summary.csv")
want = S[(S.model == "(1,0,0)(0,1,1)12+c") & (S.track == "weekdays")].set_index("kind").mape
assert abs(out["+ both"] - want["SARIMAX"]) < 0.02 and abs(out["weekdays only"] - want["SARIMA"]) < 0.02, out
print("D6 reproduces step 3")
