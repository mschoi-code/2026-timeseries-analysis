# 2026-timeseries-analysis

시계열데이터분석및실습 Team 21 프로젝트: 혼인 시차를 반영한 한국 월별 출생아 수 반등 예측

## 폴더 구성

| 폴더 | 내용 |
|---|---|
| `ols_ar2/` | 추세 + 계절 더미 회귀(OLS)와 AR(2) 오차 회귀 |
| `holt_winter/` | Holt-Winters (곱셈형, 로그 + 가산형 등 후보 비교, 공표 지연 0·2개월) |
| `var/` | VAR / VARX (출생 + 혼인 공동 예측) |
| `sarimax_etc/` | SARIMA / SARIMAX 및 추가 실험 |
| `data/` | KOSIS 원자료 (월간 인구동향 출생·혼인, 성·연령별 추계인구) |

## 공통 평가 조건

- 학습 시작: 2005.01 (2026년 값은 잠정치)
- 모델 형태·차수 확정: 2005~2015년 자료만 사용
- 평시 평가 (2016~2023): 매년 12월 기준점, 확장 윈도우로 재학습, 1~12개월 앞 예측 8회
- 반등기 평가 (2024.01~2026.07): 2023.12 시점에 계수 고정, 기준점 2023.12~2025.07에서 1~12개월 앞 예측 20회
- 공표 지연 2개월 반영: 기준점에서 2개월 전까지만 관측했다고 가정
- 지표: MAPE (주), RMSE (보조)

## 실행 방법

- `ols_ar2/`: 폴더 안에서 `python ols_ar2_regression.py` (같은 폴더의 `출생아수_혼인건수_월별.csv` 사용)
- `holt_winter/`: 폴더 안에서 각 스크립트 실행 (`../data/` 원자료 사용)
- `var/`: 폴더 안의 `README.md` 참고 (`data/monthly.csv` 포함)
- `sarimax_etc/`: 폴더 안의 `README.md` 참고. 루트의 `data/` 두 파일을 `sarimax_etc/data/raw/`에 복사한 뒤 `src/01_box_jenkins.py`, `src/02_normal_period.py` 순서로 실행

필요 패키지: `numpy`, `pandas`, `statsmodels`, `matplotlib`

## 팀

송주영, 최민서, 김준태, 문성민
