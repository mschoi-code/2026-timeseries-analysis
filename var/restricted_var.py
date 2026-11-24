"""
Restricted VAR: the births equation uses only chosen marriage lags.
  births    : const + 11 month dummies + own lags 1..P + marriage lags in MLAGS   (all in first differences of logs)
  marriages : const + 11 month dummies + own lags 1..P + births lags 1..P
Each equation is fitted by OLS. Forecasts are recursive (forecasts feed the next step), same as a VAR.
Same splits as var_model.py: normal 8 origins (expanding, refit) / rebound fixed at Dec 2023, 20 monthly origins.

Note: if every marriage lag in the births equation is >= the horizon, the births forecast never needs a
forecast of marriages (only observed values).

run:  python restricted_var.py
"""
import numpy as np
import pandas as pd
import var_model as vm

P = 11                                   # same lag order as the unrestricted VAR (AIC)
VARIANTS = {"VAR (lags 1~11)": range(1, 12),       # = unrestricted VAR-1/dummy, sanity check
            "R9 (lags 9~20)": range(9, 21),
            "R12 (lags 12~23)": range(12, 24)}


def design(D, eq, mlags, months):
    """regressor matrix for one equation; D = first differences (births, marriages)"""
    cols = {"const": np.ones(len(D))}
    for m in range(1, 12):
        cols[f"m{m}"] = (months == m).astype(float)
    own, other = ("births", "marriages") if eq == "births" else ("marriages", "births")
    for k in range(1, P + 1):
        cols[f"{own}_L{k}"] = D[own].shift(k).values
    for k in (mlags if eq == "births" else range(1, P + 1)):
        cols[f"{other}_L{k}"] = D[other].shift(k).values
    return pd.DataFrame(cols, index=D.index)


class RVAR:
    def __init__(self, L_train, mlags):
        self.mlags = list(mlags)
        D = L_train.diff().dropna()
        self.coef = {}
        for eq in ["births", "marriages"]:
            Xd = design(D, eq, self.mlags, D.index.month)
            ok = Xd.notna().all(axis=1)
            beta, *_ = np.linalg.lstsq(Xd[ok].values, D.loc[ok, eq].values, rcond=None)
            self.coef[eq] = pd.Series(beta, index=Xd.columns)

    def forecast(self, L_hist, steps):
        D = L_hist.diff().dropna().copy()
        last = L_hist.iloc[-1].copy()
        out = []
        for s in range(steps):
            d = D.index[-1] + pd.offsets.MonthBegin()
            new = {}
            for eq in ["births", "marriages"]:
                c = self.coef[eq]
                v = c["const"] + (c[f"m{d.month}"] if d.month < 12 else 0.0)
                for name, b in c.items():
                    if "_L" in name:
                        var, k = name.split("_L")
                        v += b * D[var].iloc[-int(k)]
                new[eq] = v
            D.loc[d] = new
            last = last + pd.Series(new)
            out.append((d, last["births"]))
        idx, vals = zip(*out)
        return pd.Series(np.exp(vals), index=idx)


def evaluate(L, mlags, period):
    births = np.exp(L["births"])
    if period == "normal":
        origins = [pd.Timestamp(f"{y}-12-01") for y in range(2015, 2023)]
    else:
        origins = list(pd.date_range(vm.REBOUND_FIT_END, vm.months(L.index[-1], -vm.H), freq="MS"))
        fixed = RVAR(L[:vm.months(pd.Timestamp(vm.REBOUND_FIT_END), -vm.DELAY)], mlags)
    ape, bias = [], []
    for o in origins:
        known = vm.months(o, -vm.DELAY)
        m = RVAR(L[:known], mlags) if period == "normal" else fixed
        fc = m.forecast(L[:known], vm.DELAY + vm.H).iloc[-vm.H:]
        a = births[fc.index]
        ape.append((np.abs(a - fc) / a * 100).values)
        bias.append(((fc - a) / a * 100).mean())
    ape = np.array(ape)
    return {"MAPE": ape.mean(), "h1-6": ape[:, :6].mean(), "h7-12": ape[:, 6:].mean(), "bias": np.mean(bias)}


if __name__ == "__main__":
    vm.OUTLOOK_END = pd.Timestamp(vm.OUTLOOK_END)
    raw = vm.read_csv("data/monthly.csv")
    L = vm.build_levels(raw, [])
    rows = []
    for name, ml in VARIANTS.items():
        r = {"model": name}
        for period in ["normal", "rebound"]:
            res = evaluate(L, ml, period)
            r |= {f"{period} {k}": v for k, v in res.items()}
        r["rebound growth (h7-12 / h1-6)"] = r["rebound h7-12"] / r["rebound h1-6"]
        r["normal growth (h7-12 / h1-6)"] = r["normal h7-12"] / r["normal h1-6"]
        rows.append(r)
    T = pd.DataFrame(rows).set_index("model")
    cols = ["normal MAPE", "rebound MAPE", "rebound bias", "normal h1-6", "normal h7-12",
            "rebound h1-6", "rebound h7-12", "normal growth (h7-12 / h1-6)", "rebound growth (h7-12 / h1-6)"]
    print(T[cols].round(2).T.to_string())
    out = "results" if vm.DELAY == 0 else f"results_delay{vm.DELAY}"
    import os; os.makedirs(out, exist_ok=True)
    print(f"(delay = {vm.DELAY} months)")
    T[cols].round(3).to_csv(f"{out}/restricted_var.csv")

    # which marriage lags matter in the births equation (fit 2005~2023)
    for name, ml in list(VARIANTS.items())[1:]:
        m = RVAR(L[:"2023-12"], ml)
        b = m.coef["births"][[f"marriages_L{k}" for k in ml]]
        print(f"\n{name}: marriage-lag coefficients in the births equation (2005~2023)\n", b.round(3).to_string())
