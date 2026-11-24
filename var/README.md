# TEAM 21 — VAR part

## How to run
```bash
pip install pandas numpy statsmodels matplotlib
python prepare_data.py      # raw/*.csv -> data/monthly.csv
python var_model.py         # everything -> results/
python restricted_var.py    # restricted VAR check -> results/restricted_var.csv
```
Takes a few seconds. Settings are at the top of `var_model.py` (`ECON_CANDIDATES`, `CONFIGS`, `SCENARIOS`).

Publication-lag version (data known only up to origin - 2 months) -> results_delay2/:
```bash
DELAY=2 python var_model.py            # mac / linux
$env:DELAY=2; python var_model.py      # Windows PowerShell (Remove-Item Env:DELAY to reset)
```

## Data (raw/)
| file | source | used as |
|---|---|---|
| births_marriages.csv | KOSIS 인구동향, monthly 1981.01~2026.07 (2026 = 잠정) | endogenous: births, marriages |
| population_projection.csv | KOSIS 장래인구추계 (중위), 여자 | exogenous: women 30~34 (annual, July rows, log-linear interpolation) |
| youth_employment.csv | KOSIS 경제활동인구조사 고용률, 25~29 & 30~34 (mean), not seasonally adjusted | candidate endogenous |
| mortgage_rate.csv | 한국은행 ECOS 예금은행 주택담보대출 금리 (신규취급액) | candidate endogenous |

## Pipeline (same splits as the SARIMA/SARIMAX part, PPT version, no publication lag)
1. ADF + KPSS for every series (data through 2015)
2. Screening of economic candidates: Johansen per candidate first (Granger in a VECM if rank >= 1, else in the
   differenced VAR), then keep only if Granger 5% **and** lower normal-period MAPE (rebound never used)
3. For each variable set (VAR-1, VARX-2, VARX-3): Johansen test -> VECM if rank >= 1;
   two seasonal methods: `sdiff` = (1-B)(1-B^12) / `dummy` = (1-B) + 11 month dummies;
   lag order by AIC on data through 2015; residual whiteness + stability
4. Normal period: 8 origins (Dec 2015~2022), expanding window. Rebound: coefficients fixed at Dec 2023, 20 monthly origins.
   MAPE, RMSE, bias, h=1~6 vs h=7~12
5. Interpretation (2005~2023 fit): Granger, cumulative IRF (Cholesky order: marriages before births), FEVD
6. Outlook to 2031 with the model with the lowest **normal-period** MAPE: unconditional + marriage scenarios (marriage path fixed)

## Results (MAPE %, normal / rebound)
| model | normal | rebound | rebound bias |
|---|---|---|---|
| seasonal naive | 8.51 | 7.68 | |
| VAR-1 (births, marriages), dummy, p=11 | **4.02** | 6.63 | -5.7 |
| VARX-2 (+ women 30~34), dummy, p=11 | 4.09 | **6.50** | -5.6 |
| VAR-1, sdiff, p=24 | 5.96 | 6.75 | -5.4 |
| VARX-3ref (+ youth emp + mortgage, all, reference only), VECM | 4.68 | 6.23 | -4.9 |
| *SARIMA d=1 (teammate)* | *3.59* | *7.75* | |
| *SARIMAX newlywed 12~35 (teammate)* | *3.78* | *7.59* | |
| *SARIMAX newlywed 9~32 (teammate)* | *4.06* | *5.13* | |

Full table: `results/summary.csv`.

With a 2-month publication lag (`results_delay2/summary.csv`): naive 9.92 / 8.29 (matches the SARIMA part),
VAR-1 5.88 / 8.20, VARX-2 6.09 / 7.89; SARIMA 3.96 / 9.52, SARIMAX 12~35 4.23 / 9.46, SARIMAX 9~32 4.57 / 6.00.
Same ranking; the VAR loses more from the delay than SARIMA in the normal period and is close to naive in the rebound.

## Findings
1. **Seasonality: dummies beat seasonal differencing** (normal 4.0% vs 6.0%; residuals white p=0.62 vs p=0.00).
   VAR has no MA terms, so seasonal differencing needs ~24 lags and still leaves autocorrelation.
2. **No cointegration between births and marriages** (Johansen rank 0, 2005~2015), so VAR in differences is enough; VECM not needed for the 2-variable model.
3. **Normal period:** VAR (4.0%) beats naive (8.5%) but not SARIMA (3.6%).
   **Rebound:** VAR (6.5~6.6%) beats SARIMA (7.75%) and SARIMAX 12~35 (7.59%), but not SARIMAX 9~32 (5.13%). All models forecast too low (bias about -5.5%).
4. **Error growth with horizon is NOT mainly from guessing marriages** (tested with a restricted VAR, `restricted_var.py`):
   the births equation was limited to marriage lags 12~23, so for h <= 12 it never needs a marriage forecast.
   Its error still grows more (rebound ×2.29 vs ×1.93) and it is worse overall (rebound 8.97% vs 6.63%).
   A 9~20 version is also worse (7.39%). Single monthly changes in old marriages carry little signal; the SARIMAX
   newlywed variable works because it is a smooth 24-month sum in levels, not because the values are known.
5. **Marriage -> births is clear:** Granger p < 0.0001; marriage shocks explain 32% of the 12-month births forecast error (FEVD); the cumulative response peaks about **9 months** after a marriage shock (+1.4%) and settles near +0.7%.
6. **Women 30~34 adds almost nothing** (4.02 -> 4.09, 6.63 -> 6.50).
7. **Economic variables don't help:** neither has a long-run relation with births/marriages (Johansen rank 0 for each),
   so Granger was tested in the differenced VAR: p = 0.47~0.99, and neither lowers the normal-period MAPE -> VARX-3 is empty.
   (The rank 2 in the 4-variable system comes mainly from youth employment and mortgage rate moving together: rank 1 for that pair alone.)
8. **Outlook (VAR-1, conditional):** about 285k in 2026, 266k in 2031 (unconditional); 258~265k under the marriage scenarios.

## Caveats
- The outlook's slow decline comes from the model's constant (average 2005~2026 drift). Treat it as a scenario, not a prediction.
- The marriage scenarios hold marriage **counts** (not rates) on a path; marriage changes pass through to births only weakly in a VAR in differences.
- The month-0 jump in the IRF is same-month co-movement (likely shared calendar effects), not causation.
- Women 30~34 enters with one fixed coefficient although its share of births changes over time; a fertility-weighted population offset would be cleaner (next step).
- No publication lag (PPT version), and revised KOSIS data are used.
