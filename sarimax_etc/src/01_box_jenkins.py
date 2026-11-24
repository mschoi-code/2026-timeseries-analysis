"""Step 1, Box-Jenkins on data through 2015 only: differencing tests, ACF/PACF, AIC grid, diagnostics.

Rules are in notes/decisions.md. Writes reports/01_candidates.json for step 2.
"""
import itertools
import json
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))  # appended last: scripts run with python -I
from common import BLUE, EDA_71, FIG, PROPOSAL, REP, S, SELECT_END, design, fit, label, load_features, plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf  # noqa: E402
from statsmodels.stats.diagnostic import acorr_ljungbox  # noqa: E402
from statsmodels.tsa.stattools import acf, adfuller, coint, kpss  # noqa: E402

warnings.filterwarnings("ignore")
pd.set_option("display.width", 170)
y_all, X_all = load_features()
y = y_all[:SELECT_END]
assert y.index[-1] == pd.Timestamp("2015-12-01") and len(y) == 132, "selection must stop at 2015-12"
X = X_all.loc[y.index]
TRACKS = {"proposal": False, "weekdays": True}


def job(args):
    track, spec = args
    base = dict(track=track, model=label(spec), d=spec["order"][1], spec=spec)
    try:
        r = fit(y, design(X, TRACKS[track], False), spec)
        return {**base, "k": len(r.params), "aic": r.aic, "bic": r.bic, "converged": bool(r.mle_retvals["converged"])}
    except Exception as e:  # a grid cell may fail to start; report it instead of losing the grid
        return {**base, "k": np.nan, "aic": np.nan, "bic": np.nan, "converged": False, "err": str(e)[:60]}


def diagnose(spec, calendar, with_x):
    r = fit(y, design(X, calendar, with_x), spec)
    u = pd.Series(r.resid, index=y.index).iloc[r.loglikelihood_burn:]
    n_arma = sum(spec["order"][i] for i in (0, 2)) + sum(spec["seasonal_order"][i] for i in (0, 2))
    lb = acorr_ljungbox(u, lags=[12, 24], model_df=n_arma)["lb_pvalue"].round(3).tolist()
    z = (u - u.median()) / (1.4826 * (u - u.median()).abs().median())
    return r, dict(lb12=lb[0], lb24=lb[1], aic=round(r.aic, 1), jb_p=round(float(stats.jarque_bera(u).pvalue), 3),
                   big_resid=[d.strftime("%Y-%m") for d in z[z.abs() > 3].index])


def passes(info):
    return min(info["lb12"], info["lb24"]) >= 0.05


def pick(g, calendar):
    """Lowest AIC, ties within 2 go to fewer parameters, must pass Ljung-Box; if none passes keep the first, flagged."""
    g = g[g.converged].sort_values("aic")
    near = g[g.aic < g.aic.iloc[0] + 2].sort_values(["k", "aic"])
    ranked = pd.concat([near, g[~g.model.isin(near.model)]])
    n_rejected = 0
    for _, row in ranked.iterrows():
        if passes(diagnose(row.spec, calendar, False)[1]):
            return row.spec, n_rejected
        n_rejected += 1
    return ranked.iloc[0].spec, n_rejected


if __name__ == "__main__":
    # --- differencing tests
    rows = []
    for name, s in {"log y": y, "D1": y.diff(), "D12": y.diff(S), "D1 D12": y.diff().diff(S)}.items():
        s = s.dropna()
        a = acf(s, nlags=36)
        rows.append(dict(series=name, n=len(s), adf_c=adfuller(s, regression="c", autolag="AIC")[1], adf_ct=adfuller(s, regression="ct", autolag="AIC")[1],
                         kpss_c=kpss(s, regression="c", nlags="auto")[1], kpss_ct=kpss(s, regression="ct", nlags="auto")[1],
                         acf1=a[1], acf12=a[12], acf24=a[24], acf36=a[36]))
    tests = pd.DataFrame(rows).round(3)
    tests.to_csv(REP / "01_differencing_tests.csv", index=False)
    print("B1 differencing tests (ADF H0 unit root; KPSS H0 stationary; KPSS p clipped to [.01,.10])\n", tests.to_string(index=False))

    # --- ACF / PACF of the two candidate differencings
    fig, axes = plt.subplots(2, 2, figsize=(10, 5.2), sharex=True, sharey=True)
    for i, (name, s) in enumerate({"seasonal difference (d=0, D=1)": y.diff(S), "first + seasonal difference (d=1, D=1)": y.diff().diff(S)}.items()):
        s = s.dropna()
        plot_acf(s, lags=36, ax=axes[i, 0], zero=False, title=f"ACF: {name}", color=BLUE, vlines_kwargs={"colors": BLUE})
        plot_pacf(s, lags=36, ax=axes[i, 1], zero=False, title=f"PACF: {name}", method="ywm", color=BLUE, vlines_kwargs={"colors": BLUE})
        for ax in axes[i]:
            ax.set_xticks(range(0, 37, 12))
    axes[1, 0].set_xlabel("Lag (months)")
    axes[1, 1].set_xlabel("Lag (months)")
    fig.suptitle("Log monthly births, 2005~2015", x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG / "01_acf_pacf.png")

    # --- AIC grid per track; compare AIC within one d only, likelihoods differ across differencing orders
    specs = [dict(order=(p, d, q), seasonal_order=(P, 1, Q, S), trend="c" if d == 0 else None)
             for d in (1, 0) for p, q, P, Q in itertools.product(range(3), range(3), range(2), range(2))]
    with ProcessPoolExecutor(8) as ex:
        G = pd.DataFrame(list(ex.map(job, [(t, s) for t in TRACKS for s in specs])))
    print(f"B2 grid fits {len(G)} | failed {int(G.aic.isna().sum())} | not converged {int((~G.converged).sum())}:", G[~G.converged][["track", "model"]].values.tolist())
    G.drop(columns="spec").sort_values(["track", "d", "aic"]).to_csv(REP / "01_aic_grid.csv", index=False)
    for (track, d), g in G.groupby(["track", "d"]):
        print(f"B3 {track} d={d} top 5 by AIC\n", g[g.converged].sort_values("aic").head(5)[["model", "k", "aic", "bic"]].round(1).to_string(index=False))

    # --- candidates per track: Box-Jenkins pick for each d, plus the proposal's and the 7.1% orders
    cand, diag = [], []
    for track, calendar in TRACKS.items():
        named = {}
        for d, g in G[G.track == track].groupby("d"):
            spec, n_rejected = pick(g, calendar)
            print(f"B4 {track} d={d}: pick {label(spec)} after {n_rejected} Ljung-Box rejections of {int(g.converged.sum())} converged fits")
            named.setdefault(label(spec), (f"BJ d={d}", spec))
        named.setdefault(label(PROPOSAL), ("proposal order", PROPOSAL))
        named.setdefault(label(EDA_71), ("7.1% order", EDA_71))
        for model, (source, spec) in named.items():
            r0, d0 = diagnose(spec, calendar, False)
            r1, d1 = diagnose(spec, calendar, True)
            cand.append(dict(track=track, model=model, source=source, spec=spec, sarima_lb_ok=passes(d0), sarimax_lb_ok=passes(d1)))
            b = {c: f"{r1.params[c]:+.3f} ({r1.bse[c]:.3f})" for c in ("newlywed", "women")}
            diag.append(dict(track=track, model=model, source=source, sarima_lb12=d0["lb12"], sarima_lb24=d0["lb24"], sarima_aic=d0["aic"],
                             sarimax_lb12=d1["lb12"], sarimax_lb24=d1["lb24"], sarimax_aic=d1["aic"], **b, sarimax_big_resid=d1["big_resid"]))
    D = pd.DataFrame(diag)
    D.to_csv(REP / "01_diagnostics.csv", index=False)
    print("B5 diagnostics and 2005~2015 coefficients (se)\n", D.to_string(index=False))
    (REP / "01_candidates.json").write_text(json.dumps(cand, indent=1))

    # --- does a level relation exist in the selection sample? d=0 + X leans on one
    ds = y - y.groupby(y.index.month).transform("mean")
    t, p, _ = coint(ds, X[["newlywed", "women"]], trend="c")
    print(f"B6 Engle-Granger, deseasonalized log births ~ log newlywed + log women, 2005~2015: t {t:.2f}, p {p:.3f}")
