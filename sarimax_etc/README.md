# SARIMA·SARIMAX 파트 (출생아 반등 예측)

SARIMA·SARIMAX 평가, RQ3 전망(2031), 추가 실험, 공표 지연을 넣은 실시간 평가를 담은 폴더예요.
모델 선택 규칙, 실험 선언과 결과는 `notes/decisions.md`에 날짜순으로 있어요.
`src/01~05`의 수치는 공표 지연 전 설계이고, 실시간 수치는 `reports/09_*`예요. 추정 함수를 고친 뒤 주 모델이 바뀌어서
`src/06_slides.py`는 수정 전 수치를 그려요.

## 구조

- `../data/`(레포 루트): 팀 공용 KOSIS 원본 CSV 2개. `src/common.py`의 `DATA`가 이 폴더를 읽어요. 수정 금지.
  - `월간_인구동향_출생_혼인.csv` (md5 `5a24e7af…`), 1981.01~2026.07
  - `성_및_연령별_추계인구_1세별_5세별_전국.csv` (md5 `49116752…`), 여성, 중위 추계, 1960~2035
- `data/raw/kosis_20261107/`: 11/7에 받은 KOSIS 파일 10개. 연간 출생·혼인, 연령별 출산율 1993~2025, 순위별 출생 1981~2025,
  시도별 월별 혼인 1997~2025, 연령별 혼인율, 부부 연령별 혼인. 월별 자료는 2025-12까지예요. `kosis_check.py`와 그것을 쓰는 스크립트가 읽어요.
- `data/raw/`의 다른 폴더(`kosis_api_*`, `ecos_*`, `japan_estat_*`, `team_drive_*`)는 레포에 없어요. 이 폴더를 읽는 스크립트는
  레포만으로는 돌지 않아요: `econ_var.py`, `policy_intervention.py`, `decompose_rebound.py`(KOSIS·ECOS API 자료),
  `japan_compare.py`(일본 e-Stat), `notebook_rerun.py`·`notebook_leak.py`(팀 Drive의 EDA 노트북).
  API 자료는 `scripts/fetch_kosis.py`, `scripts/fetch_ecos.py`로 다시 받을 수 있고, 각 서비스의 API 키가 필요해요.
- `src/`: SARIMA·SARIMAX 본 분석 코드. 번호 순서대로 실행해요.
- `scripts/`: 점검·추가 실험 스크립트. 각 파일 첫 줄에 무엇을 하는지와 `decisions.md`의 실험 번호가 있어요.
- `reports/`, `figures/`: `src/`와 `scripts/`를 실행하면 생기는 출력이에요. `reports/`는 레포에 올리지 않아요. 그림은 영어예요.

## 실행

`sarimax_etc/`에서 실행해요. 받은 파일을 다루므로 `-I`를 붙여요. 필요한 패키지는 pandas, numpy, statsmodels, matplotlib,
scipy이고, `ml_benchmark.py`는 scikit-learn이 더 필요해요.

```bash
python -I src/01_box_jenkins.py     # data through 2015: differencing tests, AIC grid, candidates
python -I src/02_normal_period.py   # 2016~2023: 8 origins, locks the main model
python -I src/03_rebound.py         # 2024-01~2026-07: frozen coefficients, 20 origins, decides nothing
python -I src/04_figures.py         # report figures
python -I src/05_outlook.py         # RQ3 outlook to 2031 by marriage scenario, long-horizon backtest

# exploratory checks from 2026-11-05
python -I scripts/check_births.py ../data          # data quality, proposal numbers, Granger, SARIMAX
python -I scripts/spec_search.py ../data           # 112-spec grid
python -I scripts/final_check.py ../data scripts   # attribution, placebo windows, lags, weekdays
python -I scripts/last.py ../data                  # population sums, seasonal amplitude
python -I scripts/decompose.py                     # how much of the rebound marriages explain (2026-11-06)
python -I scripts/quarterly_check.py               # monthly s=3 and quarterly s=4 against monthly s=12, report only
python -I scripts/experiments.py acefghijkbd       # experiments A~K, report only; run after src/01~03
python -I scripts/realtime.py                      # publication lag, equal blocks, MASE, DM, lag shape, scenarios; report only
python -I scripts/kosis_check.py                   # KOSIS cross-checks, population vs rate decomposition, birth order, structural effect
python -I scripts/population_variable.py           # marriage rate and age-weighted population against the current variables
python -I scripts/province_panel.py                # province panel: marriage rate leading fertility net of national shocks
python -I scripts/long_backtest.py                 # E4: 1~5-year annual errors with publication lag
python -I scripts/province_standardized.py         # E5: province panel with an age-standardized marriage rate
python -I scripts/team_summary.py                  # E6: summary tables from 09/10/14/16 plus a rerun check
```

E7 이후 스크립트(`covid_catchup.py`, `marriage_cohorts.py`, `first_analysis.py` 등)의 실행 순서와 출력 번호는
`decisions.md`의 각 실험 항목에 있어요.

제안서가 정하지 않아서 SARIMAX 파트가 고른 설정이 있어요. 30~34세 여성은 연간 값을 7월에 두고 선형 보간했어요.
이 선택이 결과에 영향을 줘요(d=1에서 선형 16.4%, 연간 계단 13.9%).

## 2026-11-05 점검 결과

모든 모델 수치는 탐색용이에요.

### 데이터

- 출생·혼인: 547개월, 결측·중복·누락 월 없음. 잠정치 표시는 2026-01~07뿐이에요.
- 인구: 30~34세 결측 없음. `80세이상` 행은 세부 구간과 겹쳐서 빼고 합산해야 `계`와 일치해요.
- 이상치(Δ1Δ12 log, 2005년 이후 robust |z|>3): 출생 7/259개월(2008-08, 2011-09, 2012-02, 2020-10, 2021-12, 2022-01 등), 혼인 1/259개월(2018-10). 원인은 조사하지 않았어요.
- 2026-08 자료가 10/28에 나왔어요. 보도 수치는 출생 23,703명, 혼인 19,218건이고 팀 파일에는 없어요. 11/10 KOSIS API 자료로 같은 값을 확인했어요(`decisions.md`).

### 제안서 수치

대부분 재현돼요. 차이는 다음과 같아요.

- Granger의 "역방향 유의하지 않음"은 차분 계열에서만 맞아요. 로그 수준에서는 양방향 모두 22/22 시차에서 유의해요.
- Granger는 시차 1..k를 묶어 검정하므로 12~35개월 창을 직접 뒷받침하지 않아요. Δ1Δ12 회귀의 개별 유의 시차는 0개월(t=3.03)과 9개월(t=2.52)뿐이고, 묶음 F는 0~11개월 p=.0007, 12~35개월 p=.028이에요.
- 슬라이드의 "projected after 2026": 2022~2072 추계라면 2023~2026년도 추계값이에요. 파일명과 팀 대화로 판단했고 PDF 본문은 못 읽었어요.
- 계절 진폭 12,500명은 어떤 기간 기준인지 찾지 못했어요.
- EDA 문서의 "2026년 46% 적게 예측"은 실제로 예측이 실측보다 31.6% 낮은 것이에요. 최종 덱에는 없어요.

### EDA의 "SARIMAX 18.0% → 7.1%"

제안서 설정(d=1, D=1)에서는 재현되지 않아요. 7.1, 18.0, 6.3%, 46%는 최종 덱에 없어요.

| 설정 | 반등기 MAPE | 평시 MAPE (8회) | 외생변수 계수 |
|---|---|---|---|
| SARIMA (0,1,1)(0,1,1)₁₂ | 18.0 | 3.59 | 없음 |
| + 혼인 시차 12 그대로 | 17.4 (EDA와 일치) | | +0.042 (se .026) |
| + 신혼부부 | 17.7 | | +0.039 (se .363) |
| + 여성 | 17.3 | | +0.384 (se .572) |
| + 신혼부부 + 여성 (제안서 Model 5) | 16.4 | 3.66 | +0.089 (p .81), +0.443 (p .45) |
| (1,0,0)(0,1,1)₁₂+c, 변수 없음 | 13.7 | 6.16 | 없음 |
| 같은 차수 + 신혼부부 | 12.6 | 5.67 | +0.95 (se .16) |
| 같은 차수 + 여성 | 11.7 | 6.91 | +0.35 (se .81) |
| 같은 차수 + 신혼부부 + 여성 | 7.1 | 5.10 | +0.97 (se .13), +0.74 (se .33) |

- EDA 노트북의 실제 설정은 미확인이에요. `log, step (2,0,0)(0,1,1)+c`도 7.08이 나와요.
- 112개 격자의 반등기 MAPE: d=1, D=1 최소 13.7, 중앙값 19.9 / d=0, D=1 최소 5.0, 중앙값 6.1 / d=0, D=0 최소 4.9, 중앙값 7.0 / d=1, D=0 최소 6.4, 중앙값 15.1.
- 10.9%p 개선 중 4.3%p는 d=0 효과, 6.6%p는 변수 효과예요.
- 매달 재학습하는 12개월 앞 예측(20회): seasonal naive 8.8(EDA 8.7), SARIMA 12.7(EDA 11.1), d=1 SARIMAX 10.2(EDA 6.3).
- 2023년 값을 반복하는 고정 시점 naive는 10.5로, d=1 모델(18.0, 16.4)보다 나아요.
- d=1에서 신호가 사라지는 이유: Δ1Δ12에서 출생과의 상관이 신혼부부 -0.026, 여성 +0.009(2006~23)예요.
- d=0의 약점: 수준 관계가 기간에 따라 달라요.

| 추정 기간 | 수준 회귀 계수 (신혼부부, 여성) | R² | Engle-Granger p |
|---|---|---|---|
| 2005~2015 | -0.35, -0.35 | .038 | .566 |
| 2005~2023 | +1.04, +1.02 | .954 | .026 |
| 2005~2026-07 | | | .041 |

- 12~35개월 창은 d=0 + 여성 설정에서 반등기와 평시 모두 가장 좋아요.

| 창 | 반등기 | 평시 | 혼인 계수 |
|---|---|---|---|
| 12~35개월 | 7.1 | 5.10 | +.97 |
| 24~47개월 | 14.7 | 5.91 | +.93 |
| 36~59개월 | 9.7 | 6.64 | -.71 |
| 48~71개월 | 14.3 | 6.99 | +.26 |
| 60~83개월 | 19.9 | 6.35 | +.70 |

### 제안서에 없는 발견

- 영업일: 월별 평일 수(공휴일 미반영)를 넣으면 SARIMA AIC가 -858.5에서 -900.1로 좋아져요. 계수 +0.232(se .034). Δ1Δ12 상관은 출생 +0.447, 혼인 +0.539이고, 이를 통제하면 출생-당월 혼인 상관이 +0.432에서 +0.253으로 줄어요.
- 인구 추계 시점: 2023년 12월 추계라 반등기에는 안전하지만, 평시 평가 시점에는 약한 미래 정보예요.
