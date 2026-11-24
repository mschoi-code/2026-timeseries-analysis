import sys
from pathlib import Path
from pypdf import PdfReader
base = Path("/mnt/c/Users/blasi/Documents/Learning/[SOI4013]Time Series Data Analysis and Practice/00_Assignment")
out = Path(sys.argv[1])
for name, tag in [("Team21_Proposal.pdf", "proposal"), ("출생아수예측_EDA.pdf", "births_eda")]:
    r = PdfReader(base / name)
    meta = r.metadata or {}
    txt = [f"# {name} pages={len(r.pages)} created={meta.get('/CreationDate')}"]
    for i, p in enumerate(r.pages, 1):
        txt.append(f"\n--- page {i} ---\n{p.extract_text()}")
    (out / f"{tag}.md").write_text("\n".join(txt), encoding="utf-8")
    print(tag, len(r.pages), sum(len(t) for t in txt))
