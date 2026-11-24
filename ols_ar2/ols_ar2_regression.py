import warnings, sys
import numpy as np, pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.stattools import pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
warnings.filterwarnings("ignore")

# 추세+계절 더미 회귀 (OLS) vs AR(2) 오차 회귀 — 제안서 평가 설계
# 평시: 2015~2022년 12월 기준점 8회, 1~12개월 앞, 확장 윈도우
# 반등기: 2023년 12월 계수 고정, 2023.12~2025.07 기준점 20회, 1~12개월 앞
# 기준점 이후 혼인은 모른다고 보고 "작년 같은 달 값"으로 채움 (9~32개월 창의 10~12개월 앞 예측에 필요)
bm = pd.read_csv("출생아수_혼인건수_월별.csv", encoding="utf-8-sig")
bm.index = pd.to_datetime(bm["연도"].astype(str) + "-" + bm["월"].astype(str) + "-01")
bm = bm.asfreq("MS")
births, marr = bm["출생아수"].astype(float), bm["혼인건수"].astype(float)
y = np.log(births)
START = pd.Timestamp("2005-01-01")
idx_all = pd.date_range(START, "2027-12-01", freq="MS")
LAST = births.index[-1]

def newlywed(m, lo):  # marriages lo ~ lo+23 months earlier
    return m.shift(lo).rolling(24).sum()

def marr_known(T):
    """marriages known at T; months after T filled with same month of last known year"""
    m = marr[:T].reindex(pd.date_range(marr.index[0], "2027-12-01", freq="MS"))
    for d in m.index[m.isna()]:
        k = d - pd.DateOffset(years=1)
        while k > T: k -= pd.DateOffset(years=1)
        m[d] = m[k]
    return m

def design(T, var):
    X = pd.DataFrame(index=idx_all)
    X["const"] = 1.0
    X["t"] = np.arange(len(idx_all)) / 120
    for mo in range(1, 12):  # December baseline
        X[f"m{mo}"] = (X.index.month == mo).astype(float)
    if var == "B": X["nw"] = np.log(newlywed(marr_known(T), 12).reindex(idx_all))
    if var == "C": X["nw"] = np.log(newlywed(marr_known(T), 9).reindex(idx_all))
    return X

def mape(a, p): a, p = np.asarray(a), np.asarray(p); return np.mean(np.abs(a - p) / a) * 100
def rmse(a, p): a, p = np.asarray(a), np.asarray(p); return np.sqrt(np.mean((a - p) ** 2))
def bias(a, p): a, p = np.asarray(a), np.asarray(p); return np.mean(p / a - 1) * 100

def snaive(target, known_end):
    k = target - pd.DateOffset(years=1)
    while k > known_end: k -= pd.DateOffset(years=1)
    return births[k]

def run(lag):
    out = {}
    # ---------- normal period: origins Dec 2015..Dec 2022, targets Jan..Dec next year
    origins = pd.date_range("2015-12-01", "2022-12-01", freq="12MS")
    recs = {k: ([], []) for k in ["naive"] + [f"{m}-{v}" for m in ["OLS", "AR2"] for v in "ABC"]}
    for O in origins:
        T = O - pd.DateOffset(months=lag)  # last known data month
        tg = pd.date_range(O + pd.DateOffset(months=1), periods=12, freq="MS")
        act = births[tg].values
        recs["naive"][0].extend(act); recs["naive"][1].extend([snaive(d, T) for d in tg])
        for v in "ABC":
            X = design(T, v); tr = slice(START, T)
            fut = pd.date_range(T + pd.DateOffset(months=1), tg[-1], freq="MS")
            b = sm.OLS(y[tr], X[tr]).fit().params
            p = np.exp(X.loc[tg] @ b)
            recs[f"OLS-{v}"][0].extend(act); recs[f"OLS-{v}"][1].extend(p)
            r = SARIMAX(y[tr], exog=X[tr], order=(2, 0, 0)).fit(disp=False)
            f = np.exp(r.forecast(len(fut), exog=X.loc[fut])).iloc[-12:]
            recs[f"AR2-{v}"][0].extend(act); recs[f"AR2-{v}"][1].extend(f)
    out["normal"] = recs
    # ---------- rebound: coefficients fixed at Dec 2023 (known through Dec 2023 - lag)
    Tfix = pd.Timestamp("2023-12-01") - pd.DateOffset(months=lag)
    params = {}
    for v in "ABC":
        X = design(Tfix, v); tr = slice(START, Tfix)
        params[v] = (sm.OLS(y[tr], X[tr]).fit().params,
                     SARIMAX(y[tr], exog=X[tr], order=(2, 0, 0)).fit(disp=False).params)
    origins = pd.date_range("2023-12-01", "2025-07-01", freq="MS")  # 20 origins
    recs = {k: ([], [], []) for k in ["naive"] + [f"{m}-{v}" for m in ["OLS", "AR2"] for v in "ABC"]}
    for O in origins:
        T = O - pd.DateOffset(months=lag)
        tg = pd.date_range(O + pd.DateOffset(months=1), periods=12, freq="MS")
        tg = tg[tg <= LAST]; act = births[tg].values
        hz = list(range(1, len(tg) + 1))
        recs["naive"][0].extend(act); recs["naive"][1].extend([snaive(d, T) for d in tg]); recs["naive"][2].extend(hz)
        for v in "ABC":
            X = design(T, v); b, pa = params[v]
            fut = pd.date_range(T + pd.DateOffset(months=1), tg[-1], freq="MS")
            recs[f"OLS-{v}"][0].extend(act); recs[f"OLS-{v}"][1].extend(np.exp(X.loc[tg] @ b)); recs[f"OLS-{v}"][2].extend(hz)
            r = SARIMAX(y[START:T], exog=X[START:T], order=(2, 0, 0)).filter(pa)
            f = np.exp(r.forecast(len(fut), exog=X.loc[fut])).iloc[-len(tg):]
            recs[f"AR2-{v}"][0].extend(act); recs[f"AR2-{v}"][1].extend(f); recs[f"AR2-{v}"][2].extend(hz)
    out["rebound"] = recs
    return out

def table(out):
    rows = {}
    for k in out["normal"]:
        a, p = out["normal"][k]; a2, p2, h2 = out["rebound"][k]
        h2 = np.array(h2); a2 = np.array(a2); p2 = np.array(p2)
        rows[k] = {"평시 MAPE": mape(a, p), "평시 RMSE": rmse(a, p), "평시 편향": bias(a, p),
                   "반등 MAPE": mape(a2, p2), "반등 RMSE": rmse(a2, p2), "반등 편향": bias(a2, p2),
                   "반등 12개월앞 MAPE": mape(a2[h2 == 12], p2[h2 == 12])}
    return pd.DataFrame(rows).T.round(2)

if __name__ == "__main__":
    # diagnostics on 2005-2023 fit
    for v in "ABC":
        X = design(pd.Timestamp("2023-12-01"), v); tr = slice(START, "2023-12-01")
        o = sm.OLS(y[tr], X[tr]).fit()
        a = SARIMAX(y[tr], exog=X[tr], order=(2, 0, 0)).fit(disp=False)
        lb_o = acorr_ljungbox(o.resid, lags=[12])["lb_pvalue"].iloc[0]
        lb_a = acorr_ljungbox(a.resid[2:], lags=[12])["lb_pvalue"].iloc[0]
        print(f"[{v}] DW OLS {durbin_watson(o.resid):.2f}  AR2 {durbin_watson(a.resid[2:]):.2f} | LB(12) p OLS {lb_o:.3f} AR2 {lb_a:.3f} | "
              f"phi {a.params['ar.L1']:.2f},{a.params['ar.L2']:.2f}" + (f" | nw coef OLS {o.params['nw']:.2f} (p={o.pvalues['nw']:.3f}) AR2 {a.params['nw']:.2f} (p={a.pvalues['nw']:.3f})" if v != "A" else ""))
        if v == "A": print("   OLS resid PACF lag1-6:", np.round(pacf(o.resid, nlags=6)[1:], 2))
    for lag in [0, 2]:
        print(f"\n===== 공표 지연 {lag}개월 =====")
        print(table(run(lag)).to_string())