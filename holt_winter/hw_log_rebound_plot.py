import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.exponential_smoothing.ets import ETSModel

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent
DATA = BASE.parent / "data" / "월간_인구동향_출생_혼인.csv"
FIG = BASE / "fig"
FIG.mkdir(exist_ok=True)

LAG = 2
HORIZON = 12
FIX_AT = pd.Timestamp("2023-12-01")
SPEC = dict(error="add", trend="add", seasonal="add", damped_trend=False, seasonal_periods=12)

raw = pd.read_csv(DATA, skiprows=2, header=None, names=["t", "birth", "marriage"])
raw["t"] = pd.PeriodIndex(raw["t"].str[:7].str.replace(".", "-"), freq="M").to_timestamp()
y = raw.set_index("t")["birth"].asfreq("MS").loc["2005":].astype(float)
ly = np.log(y)

# 2023.12 시점에 알 수 있는 자료(공표 지연 2개월이라 2023.10까지)로 계수 고정
fixed = ETSModel(ly.loc[:FIX_AT - pd.DateOffset(months=LAG)], **SPEC).fit(disp=False).params

# 기준점마다 고정 계수로 상태만 갱신하고, 기준점에서 12개월 뒤 달을 예측
points = {}
for O in pd.date_range(FIX_AT, "2025-07-01", freq="MS"):
    T = O - pd.DateOffset(months=LAG)
    res = ETSModel(ly.loc[:T], **SPEC).smooth(fixed)
    target = O + pd.DateOffset(months=HORIZON)
    h = len(pd.date_range(T + pd.DateOffset(months=1), target, freq="MS"))
    points[target] = np.exp(res.forecast(h).iloc[-1])
fc = pd.Series(points)

act = y.loc[fc.index]
print(f"12개월 앞 예측 {len(fc)}개, MAPE {np.mean(np.abs(fc / act - 1)) * 100:.2f}%, 편향 {np.mean(fc / act - 1) * 100:.2f}%")

fig, ax = plt.subplots(figsize=(10, 4.2))
ax.axvspan(FIX_AT + pd.DateOffset(days=15), y.index[-1] + pd.DateOffset(days=15), color="0.92", zorder=0)
ax.plot(y.loc["2021":] / 1e3, color="black", lw=2, label="actual")
ax.plot(fc / 1e3, color="tab:orange", marker="o", ms=3.5, lw=1.5, label="log Holt-Winters forecast (12 months ahead)")
ax.set_ylabel("monthly births (thousands)")
ax.set_title("Rebound period: coefficients fixed at Dec 2023")
ax.legend(loc="upper left", frameon=False, fontsize=9)
fig.tight_layout()
fig.savefig(FIG / "09_log_hw_rebound_12m.png", dpi=150)
