"""Step 2, normal period 2016~2023: 8 December origins, expanding window, coefficients re-estimated each origin.

Applies the selection rule in notes/decisions.md and writes reports/02_locked.json. Reads no target after 2023-12.
"""
import json
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))  # appended last: scripts run with python -I
from common import REP, TRAIN_END, design, fit, load_features, mape, rmse  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

warnings.filterwarnings("ignore")
pd.set_option("display.width", 170)
y_all, X = load_features()
y = y_all[:TRAIN_END]  # the rebound period is not loaded into this step
assert y.index[-1] == pd.Timestamp("2023-12-01")
cands = json.loads((REP / "01_candidates.json").read_text())
for c in cands:
    c["spec"] = dict(order=tuple(c["spec"]["order"]), seasonal_order=tuple(c["spec"]["seasonal_order"]), trend=c["spec"]["trend"])


def job(args):
    c, with_x, target = args
    ex = design(X, c["track"] == "weekdays", with_x)
    r = fit(y[:f"{target - 1}-12"], ex, c["spec"])
    idx = y[f"{target}"].index
    assert len(idx) == 12
    f = np.exp(r.get_forecast(12, exog=None if ex is None else ex.loc[idx]).predicted_mean)
    a = np.exp(y.loc[idx])
    out = dict(track=c["track"], model=c["model"], kind="SARIMAX" if with_x else "SARIMA", target=target, mape=mape(f, a), rmse=rmse(f, a), converged=bool(r.mle_retvals["converged"]))
    if with_x:
        out.update(b_newlywed=r.params["newlywed"], se_newlywed=r.bse["newlywed"], b_women=r.params["women"], se_women=r.bse["women"])
    return out


if __name__ == "__main__":
    targets = range(2016, 2024)
    with ProcessPoolExecutor(8) as ex:
        R = pd.DataFrame(list(ex.map(job, [(c, w, t) for c in cands for w in (False, True) for t in targets])))
    level = np.exp(y)
    naive = pd.DataFrame([dict(track="-", model="seasonal naive", kind="baseline", target=t, mape=mape(level.shift(12)[str(t)], level[str(t)]), rmse=rmse(level.shift(12)[str(t)], level[str(t)]), converged=True) for t in targets])
    R = pd.concat([naive, R], ignore_index=True)
    R.to_csv(REP / "02_normal_by_origin.csv", index=False)
    assert R.converged.all(), R[~R.converged]

    S = R.groupby(["track", "model", "kind"], sort=False).agg(mape=("mape", "mean"), rmse=("rmse", "mean"), worst=("mape", "max")).reset_index()
    S = S.merge(pd.DataFrame(cands)[["track", "model", "source", "sarima_lb_ok", "sarimax_lb_ok"]], how="left", on=["track", "model"])
    S.to_csv(REP / "02_normal_summary.csv", index=False)
    print("N1 summary, mean over 8 origins\n", S.round({"mape": 2, "rmse": 0, "worst": 2}).to_string(index=False))

    # reference values at the true optimum; the 11-05 numbers (3.59, 3.66, 6.16, 5.10) came from an optimizer that stopped early
    got = S.set_index(["track", "model", "kind"]).mape
    for key, want in {("proposal", "(0,1,1)(0,1,1)12", "SARIMA"): 3.587, ("proposal", "(0,1,1)(0,1,1)12", "SARIMAX"): 3.705,
                      ("proposal", "(1,0,0)(0,1,1)12+c", "SARIMA"): 6.107, ("proposal", "(1,0,0)(0,1,1)12+c", "SARIMAX"): 4.879}.items():
        assert abs(got[key] - want) < 0.01, (key, got[key], want)
    print("N2 reference values reproduce")

    print("N3 MAPE by target year\n", R.pivot_table(index="target", columns=["track", "model", "kind"], values="mape", sort=False).round(1).T.to_string())
    print("N4 SARIMAX coefficients by origin\n", R[R.kind == "SARIMAX"].pivot_table(index=["track", "model"], columns="target", values="b_newlywed", sort=False).round(2).to_string())
    print(R[R.kind == "SARIMAX"].pivot_table(index=["track", "model"], columns="target", values="b_women", sort=False).round(2).to_string())

    # --- declared rule: SARIMAX passing Ljung-Box through 2015, lowest MAPE; within 0.1 points the lower RMSE wins
    def best(frame):
        frame = frame.sort_values("mape")
        near = frame[frame.mape < frame.mape.iloc[0] + 0.1].sort_values("rmse")
        return near.iloc[0]

    X_ok = S[(S.kind == "SARIMAX") & (S.sarimax_lb_ok == True)].assign(d=lambda f: f.model.str[3].astype(int))  # noqa: E712
    main = best(X_ok)
    lock = dict(main=main[["track", "model", "mape", "rmse"]].to_dict(),
                best_each_d={int(d): best(g)[["track", "model", "mape", "rmse"]].to_dict() for d, g in X_ok.groupby("d")},
                best_proposal_track=best(X_ok[X_ok.track == "proposal"])[["track", "model", "mape", "rmse"]].to_dict())
    (REP / "02_locked.json").write_text(json.dumps(lock, indent=1))
    print("N5 locked", json.dumps(lock, indent=1))
