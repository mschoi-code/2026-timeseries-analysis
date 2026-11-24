import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.exponential_smoothing.ets import ETSModel

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent
DATA = BASE.parent / "data" / "월간_인구동향_출생_혼인.csv"

raw = pd.read_csv(DATA, skiprows=2, header=None, names=["t", "birth", "marriage"])
raw["t"] = pd.PeriodIndex(raw["t"].str[:7].str.replace(".", "-"), freq="M").to_timestamp()
births = raw.set_index("t")["birth"].asfreq("MS").astype(float)
y = births.loc["2005":]

# 비교할 HW 후보: (ETS 설정, 로그 변환 여부)
MODELS = {
    "덧셈형 HW  ETS(A,A,A)": (dict(error="add", trend="add", seasonal="add", damped_trend=False), False),
    "곱셈 계절 + 덧셈 오차  ETS(A,A,M)": (dict(error="add", trend="add", seasonal="mul", damped_trend=False), False),
    "곱셈형 HW  ETS(M,A,M)  [채택]": (dict(error="mul", trend="add", seasonal="mul", damped_trend=False), False),
    "곱셈형 HW + 감쇠추세  ETS(M,Ad,M)": (dict(error="mul", trend="add", seasonal="mul", damped_trend=True), False),
    "로그 + 덧셈형 HW  log-ETS(A,A,A)": (dict(error="add", trend="add", seasonal="add", damped_trend=False), True),
}
LAGS = [0, 2]


def mape(a, f):
    return np.mean(np.abs((a - f) / a)) * 100


def rmse(a, f):
    return np.sqrt(np.mean((a - f) ** 2))


def bias(a, f):
    return np.mean(f / a - 1) * 100


def build(series, spec, use_log):
    data = np.log(series) if use_log else series
    return ETSModel(data, seasonal_periods=12, **spec)


def forecast(res, h, use_log):
    fc = res.forecast(h)
    return np.exp(fc) if use_log else fc


def snaive(target, known_end):
    # 작년 같은 달이 아직 공표 전이면 한 해 더 거슬러 올라감
    k = target - pd.DateOffset(years=1)
    while k > known_end:
        k -= pd.DateOffset(years=1)
    return births[k]


def evaluate(spec, use_log, lag, origins, fixed_params=None):
    rows = []
    for O in origins:
        T = O - pd.DateOffset(months=lag)
        targets = pd.date_range(O + pd.DateOffset(months=1), periods=12, freq="MS")
        h = len(pd.date_range(T + pd.DateOffset(months=1), targets[-1], freq="MS"))
        mod = build(y.loc[:T], spec, use_log)
        res = mod.fit(disp=False) if fixed_params is None else mod.smooth(fixed_params)
        fc = forecast(res, h, use_log).iloc[-12:].values
        act = y.loc[targets].values
        rows.append((mape(act, fc), rmse(act, fc), bias(act, fc)))
    return np.array(rows).mean(axis=0)


def evaluate_snaive(lag, origins):
    rows = []
    for O in origins:
        T = O - pd.DateOffset(months=lag)
        targets = pd.date_range(O + pd.DateOffset(months=1), periods=12, freq="MS")
        act = y.loc[targets].values
        sn = np.array([snaive(d, T) for d in targets])
        rows.append((mape(act, sn), rmse(act, sn), bias(act, sn)))
    return np.array(rows).mean(axis=0)


NORMAL_ORIGINS = pd.date_range("2015-12-01", "2022-12-01", freq="12MS")
REBOUND_ORIGINS = pd.date_range("2023-12-01", "2025-07-01", freq="MS")
FIX_AT = pd.Timestamp("2023-12-01")


# 1. AIC 비교 (2005~2015, 모델 형태 확정 구간)
# 로그 모델은 자료 단위가 달라 AIC를 그대로 비교할 수 없으므로 야코비안(2 * sum(log y))을 더해 원자료 기준으로 환산
pre = y.loc[:"2015"]
aic_rows = []
for name, (spec, use_log) in MODELS.items():
    res = build(pre, spec, use_log).fit(disp=False)
    aic = res.aic + (2 * np.log(pre).sum() if use_log else 0)
    aic_rows.append({"모델": name, "AIC(2005~2015)": aic})
aic_tab = pd.DataFrame(aic_rows).set_index("모델")
aic_tab["ΔAIC"] = aic_tab["AIC(2005~2015)"] - aic_tab["AIC(2005~2015)"].min()
print("AIC 비교 (낮을수록 좋음, ΔAIC는 최솟값과의 차이)")
print(aic_tab.round(1), "\n")


# 2. 모델 x 공표 지연별 평시, 반등기 평가
rows = []
for lag in LAGS:
    for name, (spec, use_log) in MODELS.items():
        n = evaluate(spec, use_log, lag, NORMAL_ORIGINS)
        fixed = build(y.loc[:FIX_AT - pd.DateOffset(months=lag)], spec, use_log).fit(disp=False).params
        r = evaluate(spec, use_log, lag, REBOUND_ORIGINS, fixed_params=fixed)
        rows.append([lag, name, *n[:2], *r])
    n = evaluate_snaive(lag, NORMAL_ORIGINS)
    r = evaluate_snaive(lag, REBOUND_ORIGINS)
    rows.append([lag, "계절 Naive (비교 기준)", *n[:2], *r])

res_tab = pd.DataFrame(rows, columns=["공표지연(개월)", "모델", "평시 MAPE(%)", "평시 RMSE(명)",
                                      "반등기 MAPE(%)", "반등기 RMSE(명)", "반등기 편향(%)"])
res_tab = res_tab.set_index(["공표지연(개월)", "모델"])
pd.set_option("display.width", 200)
print("예측 성능 (평시 8회, 반등기 20회 평균)")
print(res_tab.round(2))

aic_tab.round(2).to_csv(BASE / "compare_aic.csv", encoding="utf-8-sig")
res_tab.round(2).to_csv(BASE / "compare_metrics_lag0_lag2.csv", encoding="utf-8-sig")
