"""
Merge the 4 raw downloads into one monthly file: data/monthly.csv
  births, marriages       KOSIS 인구동향 (2026 = preliminary 'p)')
  women30_34              KOSIS 장래인구추계 (중위, 여자, 30-34세), annual -> put in July rows (mid-year)
  youth_emp               KOSIS 경제활동인구조사 고용률, mean of 25-29 and 30-34 (not seasonally adjusted)
  mort_rate               한국은행 ECOS 예금은행 주택담보대출 금리 (신규취급액, %)
Rows run to 2035-12 so the outlook has future women30_34 values.
"""
from pathlib import Path
import pandas as pd

RAW = Path(__file__).parent / "raw"
OUT = Path(__file__).parent / "data"
OUT.mkdir(exist_ok=True)


def month_index(labels):
    s = pd.Series(labels).astype(str).str.replace("p)", "", regex=False).str.strip()
    s = s.str.replace("/", ".", regex=False).str.replace(".", "-", regex=False)
    return pd.to_datetime(s.str[:7], format="%Y-%m")


# 1) births, marriages (skip the 2nd header row "시점,전국,전국")
bm = pd.read_csv(RAW / "births_marriages.csv", encoding="utf-8-sig", skiprows=[1])
bm.index = month_index(bm.iloc[:, 0])
bm = bm.iloc[:, 1:3].astype(float)
bm.columns = ["births", "marriages"]

# 2) women 30-34, annual (3 header rows; columns = years)
pp = pd.read_csv(RAW / "population_projection.csv", encoding="utf-8-sig", header=None)
years = pp.iloc[2, 1:].astype(int).values
row = pp[pp.iloc[:, 0].astype(str).str.strip() == "30 - 34세"].iloc[0, 1:].astype(float).values
women = pd.Series(row, index=pd.to_datetime([f"{y}-07-01" for y in years]), name="women30_34")

# 3) youth employment rate (2 header rows; rows = age groups; columns = months)
ye = pd.read_csv(RAW / "youth_employment.csv", encoding="utf-8-sig", header=None)
months = month_index(ye.iloc[0, 2:].values)
ages = ye.iloc[:, 1].astype(str).str.strip()
r25 = ye[ages == "25 - 29세"].iloc[0, 2:].astype(float).values
r30 = ye[ages == "30 - 34세"].iloc[0, 2:].astype(float).values
youth = pd.Series((r25 + r30) / 2, index=months, name="youth_emp")

# 4) mortgage rate (1 header row; one data row; columns = months)
mr = pd.read_csv(RAW / "mortgage_rate.csv", encoding="utf-8-sig", header=None, dtype=str)
months = month_index(mr.iloc[0, 4:].values)
mort = pd.Series(mr.iloc[1, 4:].astype(float).values, index=months, name="mort_rate")

# merge on a full monthly index
idx = pd.date_range("1981-01-01", "2035-12-01", freq="MS")
df = pd.concat([bm, women, youth, mort], axis=1, sort=True).reindex(idx)
df.index.name = "date"
df.index = df.index.strftime("%Y-%m")
df.to_csv(OUT / "monthly.csv")

print(df.loc["2004-12":"2005-08"])
print(df.loc["2026-05":"2026-11"])
print("\nlast valid:", {c: df[c].last_valid_index() for c in df})
