import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf

warnings.filterwarnings("ignore")
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
np.random.seed(21)

plt.rcParams["font.family"] = ["Malgun Gothic", "AppleGothic", "Noto Sans CJK KR", "NanumGothic"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 110

BASE = Path(__file__).resolve().parent
DATA = BASE.parent / "data" / "월간_인구동향_출생_혼인.csv"
FIG = BASE / "fig"
FIG.mkdir(exist_ok=True)


# 1. 데이터 불러오기 (제안서대로 2005.01부터 사용, 2026년은 잠정치)
raw = pd.read_csv(DATA, skiprows=2, header=None, names=["t", "birth", "marriage"])
raw["t"] = pd.PeriodIndex(raw["t"].str[:7].str.replace(".", "-"), freq="M").to_timestamp()
y = raw.set_index("t")["birth"].asfreq("MS").loc["2005":].astype(float)
print(f"기간: {y.index[0]:%Y.%m} ~ {y.index[-1]:%Y.%m}, {len(y)}개월, 결측 {y.isna().sum()}개")


# 2. EDA: 원계열과 12개월 이동평균
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(y / 1e4, lw=1, label="월별 출생아 수")
ax.plot(y.rolling(12).mean() / 1e4, lw=2, label="12개월 이동평균")
ax.axvspan(pd.Timestamp("2024-01-01"), y.index[-1], color="tab:red", alpha=0.08, label="반등기 평가 구간")
ax.set_ylabel("만 명")
ax.set_title("월별 출생아 수 (2005.01~2026.07)")
ax.legend()
fig.tight_layout()
fig.savefig(FIG / "01_series.png")


# 3. EDA: 계절성 강도와 계절 진폭 (덧셈형 vs 곱셈형 판단용)
stl = STL(y, period=12, robust=True).fit()
season_strength = max(0, 1 - stl.resid.var() / (stl.seasonal + stl.resid).var())
trend_strength = max(0, 1 - stl.resid.var() / (stl.trend + stl.resid).var())
print(f"STL 계절성 강도 {season_strength:.2f}, 추세 강도 {trend_strength:.2f} (1에 가까울수록 강함)")

yr = y.index.year
amp = pd.DataFrame({
    "level": stl.trend.groupby(yr).mean(),
    "abs_amp": stl.seasonal.groupby(yr).agg(lambda s: s.max() - s.min()),
})
amp["rel_amp"] = amp["abs_amp"] / amp["level"] * 100
amp = amp.loc[:2025]
print(amp.round(1))

fig, ax1 = plt.subplots(figsize=(10, 4))
ax1.plot(amp.index, amp["level"] / 1e4, "o-", color="gray", label="추세 수준(만 명, 왼쪽)")
ax1.plot(amp.index, amp["abs_amp"] / 1e4, "o-", color="tab:blue", label="계절 진폭(만 명, 왼쪽)")
ax1.set_ylabel("만 명")
ax2 = ax1.twinx()
ax2.plot(amp.index, amp["rel_amp"], "s--", color="tab:orange", label="진폭/수준(%, 오른쪽)")
ax2.set_ylabel("%")
ax2.set_ylim(0, 40)
ax1.set_xticks(amp.index[::2])
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, loc="lower left")
ax1.set_title("연도별 추세 수준과 계절 진폭 (STL)")
fig.tight_layout()
fig.savefig(FIG / "02_seasonal_amplitude.png")


# 4. 모델 형태 확정: 2015년까지 데이터로만 AIC 비교 후 고정
pre = y.loc[:"2015"]
cands = {
    "덧셈 계절 HW  ETS(A,A,A)": dict(error="add", trend="add", seasonal="add", damped_trend=False),
    "곱셈 계절 HW  ETS(M,A,M)": dict(error="mul", trend="add", seasonal="mul", damped_trend=False),
    "곱셈 계절 HW + 감쇠추세  ETS(M,Ad,M)": dict(error="mul", trend="add", seasonal="mul", damped_trend=True),
}
aic = {k: ETSModel(pre, seasonal_periods=12, **v).fit(disp=False).aic for k, v in cands.items()}
print(pd.Series(aic, name="AIC (~2015)").round(1))
SPEC = cands["곱셈 계절 HW  ETS(M,A,M)"]


def fit_hw(series):
    return ETSModel(series, seasonal_periods=12, **SPEC).fit(disp=False)


def smooth_hw(series, params):
    # 계수를 다시 추정하지 않고, 고정된 계수로 상태(수준, 추세, 계절)만 새 관측값에 맞춰 갱신
    return ETSModel(series, seasonal_periods=12, **SPEC).smooth(params)


def mape(a, f):
    return np.mean(np.abs((a - f) / a)) * 100


def rmse(a, f):
    return np.sqrt(np.mean((a - f) ** 2))


# 5. 2023.12까지 학습 (반등기 평가에 쓸 고정 계수)
res23 = fit_hw(y.loc[:"2023-12"])
p = res23.params
pn = dict(zip(res23.param_names, p))
print(res23.summary().tables[1])
print(f"alpha={pn['smoothing_level']:.3f}, beta={pn['smoothing_trend']:.4f}, gamma={pn['smoothing_seasonal']:.5f}")

st = res23.states
# statsmodels의 계절지수는 평균이 1로 고정되지 않으므로, 보기 쉽게 12개월 평균 1로 정규화
seas = st["seasonal"] / st["seasonal"].rolling(12, center=True, min_periods=6).mean()
season_idx = seas.iloc[-12:].groupby(seas.index[-12:].month).mean()
print("2023년 계절지수(1=평균):", season_idx.round(3).to_dict())

fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
axes[0].plot(st["level"] / 1e4)
axes[0].set_title("수준(level, 만 명)")
axes[1].plot(st["trend"])
axes[1].axhline(0, color="k", lw=0.6)
axes[1].set_title("추세(trend, 월당 증감 명)")
axes[2].plot(seas)
axes[2].axhline(1, color="k", lw=0.6)
axes[2].set_title("계절지수(seasonal, 1 = 평균)")
fig.suptitle("곱셈형 Holt-Winters 성분 분해 (2005~2023 학습)")
fig.tight_layout()
fig.savefig(FIG / "03_components.png")


# 6. 잔차 진단
resid = res23.resid
lb = acorr_ljungbox(resid, lags=[12, 24], return_df=True)
print("Ljung-Box\n", lb.round(4))
fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
axes[0].plot(resid)
axes[0].axhline(0, color="k", lw=0.6)
axes[0].set_title("잔차 (상대오차)")
plot_acf(resid, lags=36, ax=axes[1], title="잔차 ACF")
fig.tight_layout()
fig.savefig(FIG / "04_residuals.png")


# 7. 평시 평가 (2016~2023): 매년 12월 원점, 확장 윈도우로 재학습 후 1년 앞 예측 8회
rows, normal_fc = [], []
for year in range(2015, 2023):
    origin = pd.Timestamp(f"{year}-12-01")
    train = y.loc[:origin]
    test = y.loc[origin + pd.offsets.MonthBegin(1): origin + pd.offsets.MonthBegin(12)]
    fc = fit_hw(train).forecast(12)
    snaive = train.iloc[-12:].values
    normal_fc.append(fc)
    rows.append({"원점": f"{year}.12", "HW MAPE": mape(test.values, fc.values), "HW RMSE": rmse(test.values, fc.values),
                 "SNaive MAPE": mape(test.values, snaive), "SNaive RMSE": rmse(test.values, snaive)})
normal = pd.DataFrame(rows).set_index("원점")
print("평시 평가\n", normal.round(2))
print(normal.mean().round(2))


# 8. 반등기 평가 (2024.01~2026.07): 2023.12 계수 고정, 매달 새 실측값만 넣고 1년 앞 예측 20회
rows = []
origins = pd.date_range("2023-12-01", "2025-07-01", freq="MS")
for origin in origins:
    train = y.loc[:origin]
    test = y.loc[origin + pd.offsets.MonthBegin(1): origin + pd.offsets.MonthBegin(12)]
    fc = smooth_hw(train, p).forecast(12)
    snaive = train.iloc[-12:].values
    rows.append({"원점": f"{origin:%Y.%m}", "HW MAPE": mape(test.values, fc.values), "HW RMSE": rmse(test.values, fc.values),
                 "SNaive MAPE": mape(test.values, snaive), "SNaive RMSE": rmse(test.values, snaive),
                 "HW 평균오차(%)": np.mean((fc.values - test.values) / test.values) * 100})
rebound = pd.DataFrame(rows).set_index("원점")
print("반등기 평가\n", rebound.round(2))
print(rebound.mean().round(2))

summary = pd.DataFrame({
    "평시 MAPE(%)": [normal["HW MAPE"].mean(), normal["SNaive MAPE"].mean()],
    "평시 RMSE(명)": [normal["HW RMSE"].mean(), normal["SNaive RMSE"].mean()],
    "반등기 MAPE(%)": [rebound["HW MAPE"].mean(), rebound["SNaive MAPE"].mean()],
    "반등기 RMSE(명)": [rebound["HW RMSE"].mean(), rebound["SNaive RMSE"].mean()],
}, index=["곱셈형 Holt-Winters", "계절 Naive(참고)"])
print(summary.round(1))
summary.round(2).to_csv(BASE / "metrics_summary.csv", encoding="utf-8-sig")
pd.concat({"평시": normal, "반등기": rebound}).round(3).to_csv(BASE / "metrics_by_origin.csv", encoding="utf-8-sig")


# 9. 시각화: 평시 1년 앞 예측 8회
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(y.loc["2014":"2023"] / 1e4, color="k", lw=1, label="실제")
for i, fc in enumerate(normal_fc):
    ax.plot(fc / 1e4, color="tab:blue", lw=1.5, label="HW 1년 앞 예측(매년 12월 원점)" if i == 0 else None)
ax.set_ylabel("만 명")
ax.set_title(f"평시 평가: 2016~2023 (평균 MAPE {normal['HW MAPE'].mean():.1f}%)")
ax.legend()
fig.tight_layout()
fig.savefig(FIG / "05_normal_forecasts.png")


# 10. 예측구간: 2023.12 학습 모델로 2026.07까지 예측, 95% 구간 밖으로 나간 달 세기
# 곱셈 오차 모델은 구간을 시뮬레이션으로 구하므로, 반복 수를 넉넉히 두고 시드를 고정
h = len(y.loc["2024-01":])
np.random.seed(21)
pred = res23.get_prediction(start=len(res23.data.endog), end=len(res23.data.endog) + h - 1,
                            simulate_repetitions=10000).summary_frame(alpha=0.05)
pred.index = y.loc["2024-01":].index
actual = y.loc["2024-01":]
outside = ((actual < pred["pi_lower"]) | (actual > pred["pi_upper"])).sum()
print(f"95% 예측구간 밖: {outside} / {h}개월")
yr_cmp = pd.DataFrame({"실제": actual.groupby(actual.index.year).sum(), "HW 예측": pred["mean"].groupby(pred.index.year).sum()})
yr_cmp["오차(%)"] = (yr_cmp["HW 예측"] / yr_cmp["실제"] - 1) * 100
print(yr_cmp.round(1))

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(y.loc["2021":"2023"] / 1e4, color="k", lw=1)
ax.plot(actual / 1e4, color="k", lw=1.5, label="실제")
ax.plot(pred["mean"] / 1e4, color="tab:blue", label="HW 예측 (2023.12까지 학습)")
ax.fill_between(pred.index, pred["pi_lower"] / 1e4, pred["pi_upper"] / 1e4, color="tab:blue", alpha=0.2, label="95% 예측구간")
ax.set_ylabel("만 명")
ax.set_title(f"반등기: 실제 출생아 수가 95% 예측구간 밖으로 나간 달 {outside}/{h}개월")
ax.legend(loc="upper left")
fig.tight_layout()
fig.savefig(FIG / "06_rebound_interval.png")


# 11. 반등기 1년 앞 예측 오차 추이 (원점별)
fig, ax = plt.subplots(figsize=(10, 3.5))
ax.bar(rebound.index, rebound["HW 평균오차(%)"], color="tab:blue")
ax.axhline(0, color="k", lw=0.6)
ax.set_ylabel("평균 오차(%) = (예측-실제)/실제")
ax.set_title("반등기 원점별 1년 앞 예측의 평균 오차 (음수 = 과소예측)")
ax.tick_params(axis="x", rotation=60)
fig.tight_layout()
fig.savefig(FIG / "07_rebound_bias.png")


# 12. 참고: 2026.07까지 전체로 재학습하면 추세가 어떻게 바뀌는지
res_all = fit_hw(y)
pa = dict(zip(res_all.param_names, res_all.params))
np.random.seed(21)
pred_all = res_all.get_prediction(start=len(y), end=len(y) + 29, simulate_repetitions=10000).summary_frame(alpha=0.05)
pred_all.index = pd.date_range(y.index[-1] + pd.offsets.MonthBegin(1), periods=30, freq="MS")
print(f"전체 재학습 alpha={pa['smoothing_level']:.3f}, beta={pa['smoothing_trend']:.4f}, gamma={pa['smoothing_seasonal']:.4f}")
print(f"마지막 추세(월당) 2023.12 모델: {st["trend"].iloc[-1]:.1f}명, 전체 모델: {res_all.states["trend"].iloc[-1]:.1f}명")
annual_all = pred_all["mean"].groupby(pred_all.index.year).sum()
print("전체 재학습 연간 예측(2026은 1~7월 실제 포함):", {2026: round(annual_all[2026] + y.loc["2026"].sum()), 2027: round(annual_all[2027]), 2028: round(annual_all[2028])})

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(y.loc["2019":] / 1e4, color="k", lw=1.2, label="실제")
ax.plot(pred_all["mean"] / 1e4, color="tab:red", label="HW 예측 (2026.07까지 재학습)")
ax.fill_between(pred_all.index, pred_all["pi_lower"] / 1e4, pred_all["pi_upper"] / 1e4, color="tab:red", alpha=0.15, label="95% 예측구간")
ax.set_ylabel("만 명")
ax.set_title("참고: 2026.07까지 재학습한 예측 (추세는 거의 0, 마지막 수준 유지)")
ax.legend(loc="upper left")
fig.tight_layout()
fig.savefig(FIG / "08_refit_all.png")


# 13. 참고: 로그 + 덧셈형 HW와 비교 (곱셈형과 사실상 같은 구조인지 확인)
def eval_log_add():
    spec = dict(error="add", trend="add", seasonal="add", damped_trend=False, seasonal_periods=12)
    ly = np.log(y)
    n_err = []
    for year in range(2015, 2023):
        o = pd.Timestamp(f"{year}-12-01")
        fc = np.exp(ETSModel(ly.loc[:o], **spec).fit(disp=False).forecast(12))
        n_err.append(mape(y.loc[o + pd.offsets.MonthBegin(1): o + pd.offsets.MonthBegin(12)].values, fc.values))
    pl = ETSModel(ly.loc[:"2023-12"], **spec).fit(disp=False).params
    r_err = []
    for o in origins:
        fc = np.exp(ETSModel(ly.loc[:o], **spec).smooth(pl).forecast(12))
        r_err.append(mape(y.loc[o + pd.offsets.MonthBegin(1): o + pd.offsets.MonthBegin(12)].values, fc.values))
    return np.mean(n_err), np.mean(r_err)


n_log, r_log = eval_log_add()
print(f"로그+덧셈 HW: 평시 MAPE {n_log:.2f}%, 반등기 MAPE {r_log:.2f}%")
print(f"곱셈형 HW   : 평시 MAPE {normal['HW MAPE'].mean():.2f}%, 반등기 MAPE {rebound['HW MAPE'].mean():.2f}%")
