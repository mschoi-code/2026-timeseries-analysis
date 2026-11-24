"""
TEAM 21 - VAR part: births + marriages (+ economic variables), with exogenous calendar/population terms.

INPUT CSV (one row per month, date like 2005-01 / 2005.01 / 2005-01-01)
  required : date, births, marriages
  optional : women30_34  annual projection value -> put it in the JULY row of each year (mid-year), leave other months empty
                         (add a 2004-07 row too, so Jan~Jun 2005 are interpolated instead of copied)
             holidays    number of public holidays that fall on Mon~Fri in that month (Seollal, Chuseok, ...)
             yundal      1 if the month is (mostly) a lunar leap month (윤달), else 0
             economic candidates, e.g. youth_emp, house_idx, mort_rate, ccsi
  Rows after the last births value may be added (up to OUTLOOK_END) with only women30_34 / holidays / yundal filled;
  they are needed only for the 2031 outlook.

STEPS (run: python var_model.py data.csv)
  0. Johansen cointegration test (data through ORDER_END)  -> decides whether VECM is also run
  1. Optional: screen economic candidates (Granger 5% + normal-period MAPE, data never past 2023)
  2. For each seasonal method (sdiff / dummy / vecm): lag order (AIC, data through ORDER_END), diagnostics,
     normal-period eval (8 origins), rebound eval (fixed coefficients, monthly origins), per-horizon MAPE
  3. Interpretation on the 2005~2023 fit: Granger, IRF, FEVD
  4. Outlook to OUTLOOK_END with the method that has the lowest NORMAL-period MAPE:
     unconditional + marriage scenarios (marriage path fixed)
"""
import sys
from pathlib import Path
import os
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR
from statsmodels.tsa.vector_ar.vecm import VECM, coint_johansen
from statsmodels.tsa.stattools import adfuller, kpss

warnings.filterwarnings("ignore")

# ============================== SETTINGS ==============================
CSV = sys.argv[1] if len(sys.argv) > 1 else "data/monthly.csv"
START = "2005-01"          # analysis start (proposal)
ORDER_END = "2015-12"      # lag order, Johansen test, candidate screening use data up to here
REBOUND_FIT_END = "2023-12"
OUTLOOK_END = "2031-12"
H = 12                     # 1-year-ahead
DELAY = int(os.environ.get("DELAY", "0"))   # publication lag: 0 = PPT version, 2 = data known only up to origin-2
                                             # set from the shell: DELAY=2 (mac) / $env:DELAY=2 (PowerShell)
MAXLAGS = 24               # for 2 endogenous vars; divided by (k-1) for more vars

MODES = ["sdiff", "dummy"] # sdiff: (1-B)(1-B^12), no constant | dummy: (1-B) + 11 month dummies + constant
                           # "vecm" is added automatically if Johansen finds rank >= 1 (or add it by hand)

ECON_CANDIDATES = ["youth_emp", "mort_rate"]   # screened automatically; passing ones go into VARX-3
NO_LOG = ["youth_emp", "mort_rate"]            # rates in %, not logged
# model ladder: (name, exogenous list, use screened economic vars?)
CONFIGS = [("VAR-1", [], False),
           ("VARX-2", ["women30_34"], False),
           ("VARX-3", ["women30_34"], True),          # + economic vars that pass screening
           ("VARX-3ref", ["women30_34"], "all")]      # + all candidates, for reference only (never 'best')
# calendar options also available: "weekdays", "weekends", "yundal" (needs a yundal column)

SCENARIOS = {"marriages flat": 0.00, "marriages -5%/yr": -0.05}
# =====================================================================


# ------------------------------ data ---------------------------------
def read_csv(path):
    df = pd.read_csv(path, dtype={"date": str})                      # keep "2005.10" from becoming 2005.1
    d = df["date"].str.strip().str.replace(".", "-", regex=False).str.replace(" ", "")
    df.index = pd.to_datetime(d.str[:7], format="%Y-%m")
    df = df.drop(columns="date").sort_index()
    for c in df:
        df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", "").replace({"nan": None, "": None}))
    return df.reindex(pd.date_range("2004-01-01", OUTLOOK_END, freq="MS"))   # 2004 kept for interpolation


def build_levels(raw, econ):
    """endogenous variables in (log) levels; Cholesky order = economy -> marriages -> births"""
    cols = econ + ["marriages", "births"]
    last = raw["births"].last_valid_index()
    E = raw.loc[START:last, cols].copy()
    if E.isna().any().any():
        raise ValueError(f"missing values in endogenous data:\n{E[E.isna().any(axis=1)]}")
    for c in cols:
        if c not in NO_LOG:
            E[c] = np.log(E[c])
    return E


def extend_linear(w):
    """fill leading/trailing NaNs by continuing the first/last 12-month slope (log scale)"""
    w = w.copy()
    f, l = w.first_valid_index(), w.last_valid_index()
    i_f, i_l = w.index.get_loc(f), w.index.get_loc(l)
    s_f = (w.iloc[i_f + 12] - w.iloc[i_f]) / 12
    s_l = (w.iloc[i_l] - w.iloc[i_l - 12]) / 12
    w.iloc[:i_f] = w.iloc[i_f] - s_f * np.arange(i_f, 0, -1)
    w.iloc[i_l + 1:] = w.iloc[i_l] + s_l * np.arange(1, len(w) - i_l)
    return w


def build_exog(raw, EXOG):
    """exogenous variables in levels over the whole range (to OUTLOOK_END)"""
    idx = raw.index
    X = pd.DataFrame(index=idx)
    nxt = idx + pd.offsets.MonthBegin()
    wd = np.busday_count(idx.values.astype("datetime64[D]"), nxt.values.astype("datetime64[D]"))
    hol = raw["holidays"].fillna(0).values if "holidays" in raw else 0
    if "weekdays" in EXOG:
        X["weekdays"] = wd - hol
    if "weekends" in EXOG:
        X["weekends"] = idx.days_in_month - wd
    if "yundal" in EXOG:
        if "yundal" not in raw:
            print("  ! no 'yundal' column -> yundal dropped")
        else:
            X["yundal"] = raw["yundal"].fillna(0)
    if "women30_34" in EXOG:
        if "women30_34" not in raw:
            print("  ! no 'women30_34' column -> women30_34 dropped")
        else:
            w = np.log(raw["women30_34"]).interpolate("time", limit_area="inside")
            X["women30_34"] = extend_linear(w)                       # edges: continue the nearest annual growth
    return X


def month_dummies(idx):
    D = pd.get_dummies(idx.month, prefix="m", dtype=float).set_index(idx)
    return D.drop(columns="m_12")                                      # December = baseline


# ----------------------- transforms per mode -------------------------
def tf(x, mode):
    return x.diff(12).diff(1) if mode == "sdiff" else x.diff(1)


def exog_for(X, mode):
    """transformed exogenous matrix for a mode (same index as X)"""
    Xt = tf(X, mode) if mode == "sdiff" else X.diff(1)
    if mode in ("dummy", "vecm"):
        Xt = pd.concat([month_dummies(X.index), Xt], axis=1)
    return clean_exog(Xt, mode)


NPRE = {"sdiff": 13, "dummy": 1, "vecm": 1}                           # rows lost to differencing


_SAID = set()


def clean_exog(Xt, mode):
    """drop exogenous columns that are ~constant or redundant after the transform (training rows only)"""
    T = Xt.loc[pd.Timestamp(START) + pd.DateOffset(months=NPRE[mode]):ORDER_END].fillna(0)
    keep = []
    for c in T.columns:
        cand = keep + [c]
        M = np.column_stack([np.ones(len(T))] + [T[k].values for k in cand])
        if T[c].std() > 1e-8 and np.linalg.matrix_rank(M) == M.shape[1]:
            keep.append(c)
    dropped = [c for c in T.columns if c not in keep]
    if dropped and (mode, tuple(dropped)) not in _SAID:
        _SAID.add((mode, tuple(dropped)))
        print(f"  ({mode}) dropped exogenous columns (constant/redundant after transform): {dropped}")
    return Xt[keep]


# ----------------------------- models --------------------------------
class Model:
    """one fitted model + manual forecast recursion (checked against statsmodels forecast/predict)"""

    def __init__(self, mode, L, Xt, p, rank=1):
        self.mode, self.p, self.cols = mode, p, list(L.columns)
        self.xcols = list(Xt.columns)
        if mode == "vecm":
            xe = Xt.loc[L.index].fillna(0)
            self.res = VECM(L, exog=xe, k_ar_diff=p, coint_rank=rank, deterministic="ci").fit()
        else:
            Z = tf(L, mode).iloc[NPRE[mode]:]
            xe = Xt.loc[Z.index] if self.xcols else None
            self.res = VAR(Z, exog=xe).fit(p, trend="c" if mode == "dummy" else "n")

    def forecast(self, L_hist, Xt, steps, fixed=None):
        """L_hist: levels up to the last known month. fixed: {col: Series of levels} to hold a path fixed."""
        lv = [row for row in L_hist.values]
        dates = pd.date_range(L_hist.index[-1] + pd.offsets.MonthBegin(), periods=steps, freq="MS")
        r, K = self.res, len(self.cols)
        if self.mode != "vecm":
            z = [row for row in tf(L_hist, self.mode).values[-self.p:]]
        for d in dates:
            x = Xt.loc[d, self.xcols].values.astype(float) if self.xcols else np.array([])
            if self.mode == "vecm":
                dl = [lv[-1 - i] - lv[-2 - i] for i in range(self.p)]
                step = r.alpha @ (r.beta.T @ lv[-1] + r.det_coef_coint.ravel()) + r.det_coef @ x
                step = step + sum(r.gamma[:, i * K:(i + 1) * K] @ dl[i] for i in range(self.p))
                new = lv[-1] + step
            else:
                xr = np.r_[1.0, x] if self.mode == "dummy" else x
                zt = sum(r.coefs[i] @ z[-1 - i] for i in range(self.p))
                if xr.size:
                    zt = zt + r.coefs_exog @ xr
                new = zt + lv[-1] + (lv[-12] - lv[-13] if self.mode == "sdiff" else 0)
            if fixed:
                for c, path in fixed.items():
                    new[self.cols.index(c)] = path[d]
            if self.mode != "vecm":
                zt = new - lv[-1] - (lv[-12] - lv[-13] if self.mode == "sdiff" else 0)
                z.append(zt)
            lv.append(new)
        return pd.DataFrame(np.array(lv[-steps:]), index=dates, columns=self.cols)


def choose_p(mode, L, Xt):
    K = L.shape[1]
    maxl = MAXLAGS // (K - 1)
    Ltr = L[:ORDER_END]
    Z = tf(Ltr, mode).iloc[NPRE[mode]:]
    xe = Xt.loc[Z.index] if Xt.shape[1] else None
    sel = VAR(Z, exog=xe).select_order(maxl, trend="c" if mode != "sdiff" else "n")
    p = max(int(sel.aic), 1)
    return p, sel


def johansen_rank(L, p):
    """deseasonalize levels (fixed monthly pattern, training data only) and run the trace test"""
    Ltr = L[:ORDER_END]
    ma = Ltr.rolling(12, center=True).mean().rolling(2).mean()        # 2x12 moving average
    seas = (Ltr - ma).groupby(Ltr.index.month).mean()
    seas = seas - seas.mean()
    des = Ltr - seas.loc[Ltr.index.month].values
    j = coint_johansen(des, det_order=0, k_ar_diff=p)
    rank = 0
    while rank < L.shape[1] and j.lr1[rank] > j.cvt[rank, 1]:
        rank += 1
    tab = pd.DataFrame({"H0: rank <=": range(L.shape[1]), "trace": j.lr1.round(2),
                        "5% crit": j.cvt[:, 1].round(2)})
    return min(rank, L.shape[1] - 1), tab


# --------------------------- evaluation ------------------------------
def months(ts, k):
    return ts + pd.DateOffset(months=k)


def evaluate(mode, L, Xt, p, rank, period):
    """MAPE/RMSE per origin + abs % error matrix (origins x horizon). Includes seasonal naive."""
    births = np.exp(L["births"])
    if period == "normal":
        origins = [pd.Timestamp(f"{y}-12-01") for y in range(2015, 2023)]
    else:
        origins = list(pd.date_range(REBOUND_FIT_END, months(L.index[-1], -H), freq="MS"))
        fixed_model = Model(mode, L[:months(pd.Timestamp(REBOUND_FIT_END), -DELAY)], Xt, p, rank)
    rows, ape, ape_nv = [], [], []
    for o in origins:
        known = months(o, -DELAY)
        m = Model(mode, L[:known], Xt, p, rank) if period == "normal" else fixed_model
        fc = np.exp(m.forecast(L[:known], Xt, DELAY + H)["births"]).iloc[-H:]
        a = births[fc.index]
        nv = pd.Series([births[months(t, -12)] if months(t, -12) <= known else births[months(t, -24)]
                        for t in fc.index], index=fc.index)                    # seasonal naive
        e, enav = (np.abs(a - fc) / a * 100).values, (np.abs(a - nv) / a * 100).values
        ape.append(e)
        ape_nv.append(enav)
        rows.append((f"{o:%Y-%m}", e.mean(), np.sqrt(np.mean((a - fc) ** 2)), ((fc - a) / a * 100).mean(),
                     enav.mean()))
    tab = pd.DataFrame(rows, columns=["origin", "MAPE", "RMSE", "bias", "naive MAPE"])
    return tab, np.array(ape)


# ---------------------------- screening ------------------------------
def screen(raw, X, candidates, mode="dummy"):
    L0 = build_levels(raw, [])
    Xt = exog_for(X, mode)
    p0, _ = choose_p(mode, L0, Xt)
    base = evaluate(mode, L0, Xt, p0, 1, "normal")[0].MAPE.mean()
    print(f"[screen] mode={mode}, base p={p0}, base normal MAPE={base:.2f}%")
    rows = []
    for c in candidates:
        L = build_levels(raw, [c])
        p, _ = choose_p(mode, L, Xt)
        rank, _ = johansen_rank(L, p)                                   # cointegration first
        if rank > 0:                                                   # long-run relation -> Granger in VECM
            Xv = exog_for(X, "vecm")
            m = Model("vecm", L[:ORDER_END], Xv, p, rank).res
            g_m = m.test_granger_causality(caused="marriages", causing=c).pvalue
            g_b = m.test_granger_causality(caused="births", causing=c).pvalue
            mp = evaluate("vecm", L, Xv, p, rank, "normal")[0].MAPE.mean()
        else:                                                          # none -> Granger in differenced VAR
            m = Model(mode, L[:ORDER_END], Xt, p).res
            g_m = m.test_causality("marriages", [c], kind="f").pvalue
            g_b = m.test_causality("births", [c], kind="f").pvalue
            mp = evaluate(mode, L, Xt, p, 1, "normal")[0].MAPE.mean()
        rows.append((c, p, rank, g_m, g_b, mp, mp - base))
    t = pd.DataFrame(rows, columns=["candidate", "p", "Johansen rank", "Granger->marr p", "Granger->births p",
                                    "normal MAPE", "vs base"])
    t["keep?"] = ((t["Granger->marr p"] < .05) | (t["Granger->births p"] < .05)) & (t["vs base"] < 0)
    print(t.round(4).to_string(index=False), "\n")
    return t


# ------------------------------ outlook ------------------------------
def scenario_path(L, g, end):
    """marriage COUNT path: last 12 observed months repeated, scaled by (1+g) per year"""
    last = L.index[-1]
    dates = pd.date_range(months(last, 1), end, freq="MS")
    base = np.exp(L["marriages"].iloc[-12:])
    tmpl = dict(zip(base.index.month, base.values))
    val = [tmpl[d.month] * (1 + g) ** (((d.year - last.year) * 12 + d.month - last.month) / 12) for d in dates]
    return pd.Series(np.log(val), index=dates)


def outlook(mode, L, Xt, p, rank):
    steps = (OUTLOOK_END.year - L.index[-1].year) * 12 + OUTLOOK_END.month - L.index[-1].month
    need = Xt.loc[months(L.index[-1], 1):, :]
    if need.isna().any().any():
        print("  ! exogenous values missing after the data end:", list(need.columns[need.isna().any()]),
              "-> filled with last value (add projection / calendar rows to the CSV)")
    Xf = Xt.ffill()
    m = Model(mode, L, Xf, p, rank)                                   # refit on ALL data
    hist = np.exp(L["births"])
    out = {"unconditional": np.exp(m.forecast(L, Xf, steps)["births"])}
    for name, g in SCENARIOS.items():
        path = scenario_path(L, g, OUTLOOK_END)
        out[name] = np.exp(m.forecast(L, Xf, steps, fixed={"marriages": path})["births"])
    yearly = {k: pd.concat([hist, v]).groupby(lambda d: d.year).sum() for k, v in out.items()}
    t = pd.DataFrame(yearly).loc[L.index[-1].year:]
    return (t / 1000).round(1), pd.DataFrame(out)


# ------------------------------- checks -------------------------------
def stationarity_table(L):
    """ADF (H0: unit root) and KPSS (H0: stationary) p-values, data through ORDER_END"""
    rows = []
    Ltr = L[:ORDER_END]
    for c in L:
        for name, x in [("level", Ltr[c]), ("(1-B)", Ltr[c].diff()), ("(1-B)(1-B^12)", Ltr[c].diff(12).diff())]:
            x = x.dropna()
            rows.append((c, name, adfuller(x, autolag="AIC")[1], kpss(x, regression="c", nlags="auto")[1]))
    t = pd.DataFrame(rows, columns=["variable", "transform", "ADF p", "KPSS p"])
    t["verdict"] = np.where((t["ADF p"] < .05) & (t["KPSS p"] > .05), "stationary",
                   np.where((t["ADF p"] >= .05) & (t["KPSS p"] <= .05), "non-stationary", "conflict"))
    return t


# ------------------------------- plots --------------------------------
def plot_rebound(L, models, path):
    """models: (label, mode, Xt, p, rank). Coefficients are fit on data known at Dec 2023 (= Dec 2023 - DELAY)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    births = np.exp(L["births"])
    fit_end = months(pd.Timestamp(REBOUND_FIT_END), -DELAY)
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.plot(births["2021":] / 1e3, color="black", lw=2, label="actual")
    for (label, mode, Xt, p, rank), color in zip(models, ["tab:orange", "tab:blue", "tab:green", "tab:red"]):
        m = Model(mode, L[:fit_end], Xt, p, max(rank, 1))
        fc = np.exp(m.forecast(L[:fit_end], Xt, len(L[fit_end:]) - 1)["births"])
        ax.plot(fc[REBOUND_FIT_END:] / 1e3, color=color, lw=1.5, ls="--",
                label=f"{label}: one forecast from Dec 2023 (data to {fit_end:%b %Y})")
        pts = {}
        for o in pd.date_range(REBOUND_FIT_END, months(L.index[-1], -H), freq="MS"):
            known = months(o, -DELAY)
            f = np.exp(m.forecast(L[:known], Xt, DELAY + H)["births"])
            pts[f.index[-1]] = f.iloc[-1]
        ax.plot(pd.Series(pts) / 1e3, color=color, lw=1.5, marker="o", ms=3,
                label=f"{label}: 12-month-ahead, monthly origins")
    ax.axvspan(pd.Timestamp("2024-01-01"), L.index[-1], color="0.94")
    ax.set_ylabel("monthly births (thousands)")
    ttl = "Rebound period: coefficients fixed at Dec 2023"
    ax.set_title(ttl + (f" (publication lag {DELAY} months)" if DELAY else ""))
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_outlook(yearly, hist_years, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(hist_years / 1e3, color="black", lw=2, marker="o", ms=3, label="actual")
    for c in yearly:
        ax.plot(yearly[c], marker="o", ms=3, ls="--", label=c)
    ax.set_ylabel("annual births (thousands)")
    ax.set_title("Outlook (2026 = actual Jan~Jul + forecast)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ------------------------------- main --------------------------------
def run_config(name, L, X):
    """Johansen + all seasonal modes for one variable set. Returns summary rows and fitted 2023 models."""
    print(f"\n#################### {name}: endogenous {list(L.columns)}, exogenous {list(X.columns)} ####################")
    p_d, _ = choose_p("dummy", L, exog_for(X, "dummy"))
    rank, jt = johansen_rank(L, p_d)
    print("[Johansen trace test, data through", ORDER_END, "]\n", jt.to_string(index=False))
    print(f"-> rank = {rank}" + (" -> VECM added" if rank > 0 else " -> no long-run relation, VAR in differences only"))
    modes = MODES + (["vecm"] if rank > 0 else [])
    rows, fits = [], {}
    for mode in modes:
        Xt = exog_for(X, mode)
        pm = "dummy" if mode == "vecm" else mode
        p, sel = choose_p(pm, L, exog_for(X, pm))
        m23 = Model(mode, L[:REBOUND_FIT_END], Xt, p, max(rank, 1))
        white = m23.res.test_whiteness(nlags=max(24, p + 2)).pvalue
        stable = m23.res.is_stable() if mode != "vecm" else "-"
        row = {"config": name, "mode": mode, "p": p, "BIC p": sel.bic, "whiteness p": white, "stable": stable}
        for period in ["normal", "rebound"]:
            tab, ape = evaluate(mode, L, Xt, p, max(rank, 1), period)
            row |= {f"{period} MAPE": tab.MAPE.mean(), f"{period} RMSE": tab.RMSE.mean(),
                    f"{period} bias": tab.bias.mean(),
                    f"{period} h1-6": ape[:, :6].mean(), f"{period} h7-12": ape[:, 6:].mean()}
            naive = tab["naive MAPE"].mean()
        rows.append(row)
        fits[mode] = (m23, Xt, p, rank)
        print(f"  {mode:6s} p={p:2d}  normal MAPE {row['normal MAPE']:.2f}%  rebound MAPE {row['rebound MAPE']:.2f}%"
              f"  (whiteness p={white:.3f}, stable={stable})")
    return rows, fits


if __name__ == "__main__":
    OUTLOOK_END = pd.Timestamp(OUTLOOK_END)
    out = Path("results" if DELAY == 0 else f"results_delay{DELAY}")
    out.mkdir(exist_ok=True)
    raw = read_csv(CSV)
    print(f"data: {CSV} | delay = {DELAY} months")

    # stationarity of every candidate series
    L_all = build_levels(raw, ECON_CANDIDATES)
    st = stationarity_table(L_all)
    print("\n[ADF / KPSS, data through", ORDER_END, "]\n", st.round(3).to_string(index=False))
    st.round(4).to_csv(out / "stationarity.csv", index=False)

    # screening economic candidates on top of VARX-2 (women30_34)
    print("\n[screening economic candidates, base = births + marriages + women30_34]")
    scr = screen(raw, build_exog(raw, ["women30_34"]), ECON_CANDIDATES)
    scr.round(4).to_csv(out / "screening.csv", index=False)
    keep = list(scr.loc[scr["keep?"], "candidate"])
    print("-> kept for VARX-3:", keep or "none")

    # ladder
    all_rows, all_fits, data = [], {}, {}
    for name, exog, use_econ in CONFIGS:
        econ = ECON_CANDIDATES if use_econ == "all" else (keep if use_econ else [])
        if use_econ and not econ:
            print(f"\n{name}: no economic variable passed screening -> skipped")
            continue
        L = build_levels(raw, econ)
        X = build_exog(raw, exog)
        rows, fits = run_config(name, L, X)
        all_rows += rows
        all_fits[name] = fits
        data[name] = (L, X)

    S = pd.DataFrame(all_rows)
    naive_rows = []
    L0 = data["VAR-1"][0]
    for period in ["normal", "rebound"]:
        tab, _ = evaluate("dummy", L0, exog_for(build_exog(raw, []), "dummy"), 1, 1, period)
        naive_rows.append(tab["naive MAPE"].mean())
    print("\n==================== SUMMARY (MAPE %) ====================")
    show = ["config", "mode", "p", "normal MAPE", "rebound MAPE", "rebound bias", "normal h1-6", "normal h7-12",
            "rebound h1-6", "rebound h7-12", "normal RMSE", "rebound RMSE", "whiteness p", "stable"]
    print(S[show].round(2).to_string(index=False))
    print(f"seasonal naive: normal {naive_rows[0]:.2f}%  rebound {naive_rows[1]:.2f}%")
    print("bias = mean (forecast - actual) / actual in %; negative = forecasts too low")
    S.round(3).to_csv(out / "summary.csv", index=False)

    # best = lowest NORMAL-period MAPE (rebound is never used for choosing)
    b = S.loc[S.loc[S.config != "VARX-3ref", "normal MAPE"].idxmin()]
    bname, bmode = b["config"], b["mode"]
    L, X = data[bname]
    m23, Xt, p, rank = all_fits[bname][bmode]
    print(f"\nbest by normal-period MAPE: {bname} / {bmode}")

    # interpretation on the best VAR-in-differences mode of the best config
    imode = bmode if bmode != "vecm" else "dummy"
    r = all_fits[bname][imode][0].res
    cols = list(L.columns)
    i, j = cols.index("marriages"), cols.index("births")
    print(f"\n[interpretation: {bname} / {imode}, fit 2005~2023]")
    for c in cols[:-1]:
        g = r.test_causality("births", [c], kind="f")
        print(f"Granger {c} -> births: F = {g.test_statistic:.2f}, p = {g.pvalue:.4f}")
    for c in cols[:-2]:
        g = r.test_causality("marriages", [c], kind="f")
        print(f"Granger {c} -> marriages: F = {g.test_statistic:.2f}, p = {g.pvalue:.4f}")
    irf = r.irf(24)
    fe = r.fevd(24).decomp[j]
    fevd = pd.DataFrame(fe[[0, 5, 11, 23]] * 100, index=["h=1", "h=6", "h=12", "h=24"], columns=cols).round(1)
    print("FEVD of births (% of forecast error variance from each variable's shocks):\n", fevd.to_string())
    fevd.to_csv(out / "fevd_births.csv")
    # cumulative orthogonalized response = effect on the LEVEL of log births (x100 = %)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cum = irf.orth_cum_effects[:, j, i] * 100
    lo, hi = irf.cum_effect_conf_int(orth=True) if hasattr(irf, "cum_effect_conf_int") else (None, None)
    sd = r.irf_errband_mc(orth=True, repl=500, steps=24, seed=0, cum=True) if lo is None else None
    if sd is not None:
        lo, hi = sd
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(25), cum, color="tab:blue", lw=2, label="cumulative response")
    ax.fill_between(range(25), lo[:, j, i] * 100, hi[:, j, i] * 100, color="tab:blue", alpha=.15, label="95% band")
    ax.axhline(0, color="black", lw=.8)
    ax.set_xlabel("months after a 1-s.d. marriage shock")
    ax.set_ylabel("births level response (%)")
    ax.set_title(f"Marriage shock -> births ({bname}/{imode}, fit 2005~2023)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "irf_marriages_to_births.png", dpi=130)
    plt.close(fig)
    print("cumulative births response to a marriage shock (%), months 0,3,6,9,12,18,24:",
          [round(float(cum[k]), 2) for k in (0, 3, 6, 9, 12, 18, 24)])

    # plots: rebound
    pick = [("VAR-1", "dummy"), (bname, bmode)]
    models = [(f"{n}/{md}", md, all_fits[n][md][1], all_fits[n][md][2], all_fits[n][md][3])
              for n, md in dict.fromkeys(pick)]
    plot_rebound(L, models, out / "rebound_forecasts.png")

    # outlook
    print(f"\n[outlook to {OUTLOOK_END:%Y-%m}: {bname} / {bmode}, refit on all data]")
    yearly, monthly = outlook(bmode, L, Xt, p, max(rank, 1))
    print(yearly.to_string())
    yearly.to_csv(out / "outlook_yearly.csv")
    monthly.round(0).to_csv(out / "outlook_monthly.csv")
    hist = np.exp(L["births"]).groupby(L.index.year).sum()
    plot_outlook(yearly, hist.loc[2015:L.index[-1].year - 1], out / "outlook.png")
    print(f"\nsaved to {out}/: summary.csv, stationarity.csv, screening.csv, fevd_births.csv, "
          "outlook_*.csv, rebound_forecasts.png, irf_marriages_to_births.png, outlook.png")
