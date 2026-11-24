import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.exponential_smoothing.ets import ETSModel

warnings.filterwarnings("ignore")

# 공표 지연 2개월 반영판: 기준점 O에서는 O-2개월까지만 관측했다고 보고, 예측 대상 달(O+1 ~ O+12)은 그대로 둠
LAG = 2
BASE = Path(__file__).resolve().parent
DATA = BASE.parent / "data" / "월간_인구동향_출생_혼인.csv"

raw = pd.read_csv(DATA, skiprows=2, header=None, names=["t", "birth", "marriage"])
raw["t"] = pd.PeriodIndex(raw["t"].str[:7].str.replace(".", "-"), freq="M").to_timestamp()
births = raw.set_index("t")["birth"].asfreq("MS").astype(float)
y = births.loc["2005":]

SPEC = dict(error="mul", trend="add", seasonal="mul", damped_trend=False, seasonal_periods=12)


def mape(a, f):
    return np.mean(np.abs((a - f) / a)) * 100


def rmse(a, f):
    return np.sqrt(np.mean((a - f) ** 2))


def bias(a, f):
    return np.mean(f / a - 1) * 100


def snaive(target, known_end):
    # 작년 같은 달이 아직 공표 전이면 한 해 더 거슬러 올라감
    k = target - pd.DateOffset(years=1)
    while k > known_end:
        k -= pd.DateOffset(years=1)
    return births[k]


def evaluate(origins, fixed_params=None):
    rows = []
    for O in origins:
        T = O - pd.DateOffset(months=LAG)
        targets = pd.date_range(O + pd.DateOffset(months=1), periods=12, freq="MS")
        h = len(pd.date_range(T + pd.DateOffset(months=1), targets[-1], freq="MS"))
        if fixed_params is None:
            res = ETSModel(y.loc[:T], **SPEC).fit(disp=False)
        else:
            res = ETSModel(y.loc[:T], **SPEC).smooth(fixed_params)
        fc = res.forecast(h).iloc[-12:].values
        act = y.loc[targets].values
        sn = np.array([snaive(d, T) for d in targets])
        rows.append({"기준점": f"{O:%Y.%m}", "HW MAPE": mape(act, fc), "HW RMSE": rmse(act, fc), "HW 편향": bias(act, fc),
                     "SNaive MAPE": mape(act, sn), "SNaive RMSE": rmse(act, sn)})
    return pd.DataFrame(rows).set_index("기준점")


# 평시: 2015~2022년 매년 12월 기준점, 매번 재학습
normal = evaluate(pd.date_range("2015-12-01", "2022-12-01", freq="12MS"))

# 반등기: 2023.12 시점에 알 수 있는 자료(2023.10까지)로 계수 고정, 기준점 2023.12~2025.07
fixed = ETSModel(y.loc[:pd.Timestamp("2023-12-01") - pd.DateOffset(months=LAG)], **SPEC).fit(disp=False).params
rebound = evaluate(pd.date_range("2023-12-01", "2025-07-01", freq="MS"), fixed_params=fixed)

print("평시\n", normal.round(2))
print("반등기\n", rebound.round(2))
summary = pd.DataFrame({
    "평시 MAPE(%)": [normal["HW MAPE"].mean(), normal["SNaive MAPE"].mean()],
    "평시 RMSE(명)": [normal["HW RMSE"].mean(), normal["SNaive RMSE"].mean()],
    "반등기 MAPE(%)": [rebound["HW MAPE"].mean(), rebound["SNaive MAPE"].mean()],
    "반등기 RMSE(명)": [rebound["HW RMSE"].mean(), rebound["SNaive RMSE"].mean()],
    "반등기 편향(%)": [rebound["HW 편향"].mean(), np.nan],
}, index=["곱셈형 Holt-Winters", "계절 Naive"])
print(summary.round(2))
print("반등기 과소예측 기준점 수:", (rebound["HW 편향"] < 0).sum(), "/", len(rebound))
summary.round(2).to_csv(BASE / "metrics_summary_lag2.csv", encoding="utf-8-sig")
