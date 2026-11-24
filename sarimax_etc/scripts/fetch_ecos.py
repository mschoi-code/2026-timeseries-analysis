"""Download monthly Bank of Korea ECOS series for the economic block (rates, loans, house prices, CPI, jobs).

Reads the key from ~/.config/api_keys/ecos and never prints it or a request URL.
Writes data/raw/ecos_<today>/ecos_monthly.csv (long format). Run from sarimax_etc/: python -I scripts/fetch_ecos.py
"""
import json
import time
import urllib.request
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
KEY = (Path.home() / ".config/api_keys/ecos").read_text().strip()
END = date.today().strftime("%Y%m")
REB = {"901Y093": "H69A", "901Y149": "00"}  # Korea Real Estate Board price index by region, all housing; base 2021.06 (2003-11~2025-03) and 2026.06 (2021-06~)
SERIES = [  # (name, stat code, item codes)
    ("base_rate", "722Y001", ["0101000"]),
    ("loan_rate_household_new", "121Y006", ["BECBLA03"]),
    ("loan_rate_mortgage_new", "121Y006", ["BECBLA0302"]),
    ("household_loans_depository", "151Y005", ["1110000"]),
    ("household_loans_housing", "151Y005", ["11100A0"]),
    ("policy_mortgage_loans_hf_hug", "151Y005", ["1120093"]),
    ("jeonse_loans_banks", "151Y005", ["1120094"]),
    ("house_price_kb_national", "901Y062", ["P63A"]),
    ("house_price_kb_seoul", "901Y062", ["P63AD"]),
    ("cpi", "901Y009", ["0"]),
    ("employment_rate", "901Y027", ["I61E"]),
    ("unemployment_rate", "901Y027", ["I61BC"]),
]


def call(path, what):
    try:
        with urllib.request.urlopen(f"https://ecos.bok.or.kr/api/{path}", timeout=60) as r:
            return json.load(r)
    except Exception as e:
        raise SystemExit(f"{what}: request failed ({type(e).__name__})")  # the URL carries the key


def regions(stat):
    rows = call(f"StatisticItemList/{KEY}/json/kr/1/500/{stat}", stat)["StatisticItemList"]["row"]
    return sorted({r["ITEM_CODE"] for r in rows if r["GRP_NAME"] == "계정항목"})


def search(stat, items):
    d = call(f"StatisticSearch/{KEY}/json/kr/1/100000/{stat}/M/190001/{END}/" + "/".join(items), f"{stat} {items}")
    if "StatisticSearch" not in d:
        raise SystemExit(f"{stat} {items}: {d.get('RESULT', {}).get('CODE')} {d.get('RESULT', {}).get('MESSAGE')}")
    return d["StatisticSearch"]["row"]


def main():
    out = ROOT / "data" / "raw" / f"ecos_{date.today():%Y%m%d}"
    out.mkdir(parents=True, exist_ok=True)
    frames = []
    series = SERIES + [(f"house_price_reb_{stat}_{r}", stat, [kind, r]) for stat, kind in REB.items() for r in regions(stat)]
    for name, stat, items in series:
        rows = search(stat, items)
        f = pd.DataFrame(rows)
        frames.append(pd.DataFrame({"series": name, "stat_code": stat, "item": "/".join(items),
                                    "item_name": (f.ITEM_NAME1 + " " + f.get("ITEM_NAME2", pd.Series("", index=f.index)).fillna("")).str.strip(),
                                    "unit": f.UNIT_NAME, "month": f.TIME, "value": pd.to_numeric(f.DATA_VALUE, errors="coerce")}))
        time.sleep(0.3)  # be gentle with the API
    D = pd.concat(frames, ignore_index=True)
    dup = D[D.duplicated(["series", "item_name", "month"], keep=False)]  # some tables come as original and seasonally adjusted
    assert dup.empty, f"duplicate months: {dup.groupby('series').item_name.unique().to_dict()}"
    D.to_csv(out / "ecos_monthly.csv", index=False, encoding="utf-8")
    S = D.groupby(["series", "item_name"], sort=False).agg(unit=("unit", "first"), start=("month", "min"), end=("month", "max"), n=("value", "size"), missing=("value", lambda v: int(v.isna().sum())))
    S.to_csv(out / "series_summary.csv", encoding="utf-8")
    pd.set_option("display.width", 200)
    print(S.to_string())
    print(f"\nsaved {len(D)} rows to {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
