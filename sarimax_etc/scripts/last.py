import sys
from pathlib import Path
import numpy as np, pandas as pd
raw = Path(sys.argv[1])
p = pd.read_csv(raw / "성_및_연령별_추계인구_1세별_5세별_전국.csv", header=None, dtype=str)
pop = p.iloc[3:].set_index(0); pop.columns = p.iloc[2, 1:].astype(int).values
pop = pop.replace("-", np.nan).astype(float)
five = [i for i in pop.index if i not in ("계", "80세이상")]
print("pop sum(five)-total, max abs, 2000+:", (pop.loc[five].sum() - pop.loc["계"]).loc[2000:].abs().max())
print("80+ detail sums to 80세이상, max abs, 2000+:", (pop.loc[["80 - 84세", "85 - 89세", "90 - 94세", "95 - 99세", "100세 이상"]].sum() - pop.loc["80세이상"]).loc[2000:].abs().max())
print("NaN in 30-34:", int(pop.loc["30 - 34세"].isna().sum()), "| 1990-1999 total check:", (pop.loc[[i for i in five if i not in ("80 - 84세","85 - 89세","90 - 94세","95 - 99세","100세 이상")]+["80세이상"]].sum() - pop.loc["계"]).abs().max())
m = pd.read_csv(raw / "월간_인구동향_출생_혼인.csv", skiprows=2, header=None, names=["t", "b", "m"], dtype={"t": str})
m.index = pd.to_datetime(m.t.str.slice(0, 7), format="%Y.%m")
print("12m avg 2017-01:", round(m.b.rolling(12).mean()["2017-01-01"]))
r = m.b / m.b.rolling(12, center=True).mean().rolling(2).mean().shift(-1)
for a, b in [("1995","2004"),("1997","2001"),("2000","2009"),("1998","2002"),("2005","2014"),("2008","2013")]:
    s = r[a:b].groupby(r[a:b].index.month).mean(); print(a, b, "abs %.0f rel %.1f%%" % ((s.max()-s.min())*m.b[a:b].mean(), (s.max()-s.min())*100))
