"""Download KOSIS tables through the OpenAPI, every item and class, chunked under the 40,000-cell limit per call.

Reads the key from ~/.config/api_keys/kosis and never prints it or a request URL.
Writes data/raw/kosis_api_<today>/<name>.csv (rows as KOSIS returns them). Run from births/: python -I scripts/fetch_kosis.py
"""
import json
import time
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
KEY = (Path.home() / ".config/api_keys/kosis").read_text().strip()
TABLES = [  # (name, table id, period label as KOSIS meta gives it)
    ("births_by_marriage_duration", "DT_1B81A08", "년"),
    ("births_by_legal_marital_status", "DT_1B81A16", "년"),
    ("mean_age_at_birth_by_order", "DT_1B81A20", "년"),
    ("newlywed_months_to_first_birth", "DT_1NW2019", "년"),
    ("vital_events_monthly_by_province", "DT_1B8000G", "월"),
    ("labor_force_by_sex_age_monthly", "DT_1DA7012S", "월"),
    ("views_on_marriage", "DT_1SSFA060R", "2년"),
]
CODE = {"년": "Y", "월": "M", "2년": "F"}
MAX_CELLS = 39000


def call(service, params, what):
    q = urllib.parse.urlencode(dict(params, apiKey=KEY, format="json", jsonVD="Y"))
    try:
        with urllib.request.urlopen(f"https://kosis.kr/openapi/{service}?{q}", timeout=120) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception as e:
        raise SystemExit(f"{what}: request failed ({type(e).__name__})")  # the URL carries the key


def periods(start, end, se):
    if se == "월":
        return [p.strftime("%Y%m") for p in pd.period_range(start.replace(".", "-"), end.replace(".", "-"), freq="M")]
    return [str(y) for y in range(int(start[:4]), int(end[:4]) + 1)]


def fetch(name, tbl, se):
    itm = call("statisticsData.do", dict(method="getMeta", type="ITM", orgId="101", tblId=tbl), f"{tbl} meta")
    sizes = {}
    for r in itm:
        sizes[r["OBJ_ID"]] = sizes.get(r["OBJ_ID"], 0) + 1
    classes = [o for o in sizes if o != "ITEM"]  # order of appearance = objL1, objL2, ...
    cells = sizes.get("ITEM", 1)
    for o in classes:
        cells *= sizes[o]
    prd = next(p for p in call("statisticsData.do", dict(method="getMeta", type="PRD", orgId="101", tblId=tbl), f"{tbl} periods") if p["PRD_SE"] == se)
    span = periods(prd["STRT_PRD_DE"], prd["END_PRD_DE"], se)
    step = max(1, MAX_CELLS // cells)
    rows = []
    for i in range(0, len(span), step):
        part = span[i:i + step]
        params = dict(method="getList", orgId="101", tblId=tbl, itmId="ALL", prdSe=CODE[se], startPrdDe=part[0], endPrdDe=part[-1])
        params |= {f"objL{k + 1}": "ALL" for k in range(len(classes))}
        d = call("Param/statisticsParameterData.do", params, f"{tbl} {part[0]}~{part[-1]}")
        if isinstance(d, dict):  # an error object; "no data" for a gap in a multi-year survey is fine
            if str(d.get("err")) == "30":
                continue
            raise SystemExit(f"{tbl} {part[0]}~{part[-1]}: {d.get('err')} {d.get('errMsg')}")
        rows += d
        time.sleep(0.3)  # be gentle with the API
    D = pd.DataFrame(rows)
    assert len(D), f"{tbl}: no rows"
    return D, dict(name=name, table=tbl, period=se, start=D.PRD_DE.min(), end=D.PRD_DE.max(), rows=len(D), cells_per_period=cells, calls=-(-len(span) // step))


def main():
    out = ROOT / "data" / "raw" / f"kosis_api_{date.today():%Y%m%d}"
    out.mkdir(parents=True, exist_ok=True)
    summary = []
    for name, tbl, se in TABLES:
        D, info = fetch(name, tbl, se)
        D.to_csv(out / f"{name}.csv", index=False, encoding="utf-8")
        summary.append(info)
    S = pd.DataFrame(summary)
    S.to_csv(out / "tables_summary.csv", index=False, encoding="utf-8")
    pd.set_option("display.width", 200)
    print(S.to_string(index=False))
    print(f"\nsaved to {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
