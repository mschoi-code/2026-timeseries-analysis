"""Step 4, slide figures from the saved step 2 and 3 tables. English labels: the course runs in English."""
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))  # appended last: scripts run with python -I
from common import BLUE, FIG, INK, MUTED, ORANGE, REP, load_monthly, plt  # noqa: E402
import pandas as pd  # noqa: E402

LOCK = json.loads((REP / "02_locked.json").read_text())
M1, M0 = LOCK["best_each_d"]["1"], LOCK["best_each_d"]["0"]
D1 = f"{M1['model']}|{M1['track']}|SARIMAX"
D0 = f"{M0['model']}|{M0['track']}|SARIMAX"

# --- rebound: one path from 2023-12 against actual births
O = pd.read_csv(REP / "03_rebound_oneshot.csv", parse_dates=["target"])
actual = load_monthly().births["2022":] / 1000
fig, ax = plt.subplots(figsize=(9, 4.2))
ax.plot(actual.index, actual, color=INK, label="Actual births")
for name, color, text in ((D1, BLUE, "d=1 SARIMAX (main)"), (D0, ORANGE, "d=0 SARIMAX (alternative)")):
    g = O[O.name == name]
    ax.plot(g.target, g.forecast / 1000, color=color, label=text)
    ax.annotate(text, (g.target.iloc[-1], g.forecast.iloc[-1] / 1000), xytext=(6, 0), textcoords="offset points", fontsize=9, va="center")
ax.annotate("Actual", (actual.index[-1], actual.iloc[-1]), xytext=(6, 0), textcoords="offset points", fontsize=9, va="center")
ax.axvline(pd.Timestamp("2023-12-01"), color=MUTED, lw=0.8, ls="--")
ax.text(pd.Timestamp("2023-12-15"), 29.5, "coefficients and data frozen at 2023-12", fontsize=8.5, color=INK)
ax.set_ylabel("Monthly births (thousands)")
ax.set_ylim(14, 30)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3)
ax.set_title("Both models under-forecast the rebound; only d=0 turns upward", loc="left", fontsize=12)
fig.savefig(FIG / "03_rebound_paths.png")

# --- normal vs rebound error for the declared models
S = pd.read_csv(REP / "03_rebound_summary.csv")
ROWS = [("seasonal naive", "-", "baseline", "Seasonal naive"),
        ("(0,1,1)(0,1,1)12", "proposal", "SARIMA", "Proposal order, SARIMA"), ("(0,1,1)(0,1,1)12", "proposal", "SARIMAX", "Proposal order, SARIMAX"),
        (M1["model"], M1["track"], "SARIMA", "d=1 main + weekdays, SARIMA"), (M1["model"], M1["track"], "SARIMAX", "d=1 main + weekdays, SARIMAX"),
        (M0["model"], M0["track"], "SARIMA", "d=0 alternative, SARIMA"), (M0["model"], M0["track"], "SARIMAX", "d=0 alternative, SARIMAX")]
vals = [S[(S.model == m) & (S.track == t) & (S.kind == k)].iloc[0] for m, t, k, _ in ROWS]
fig, ax = plt.subplots(figsize=(9, 4.2))
for i, (v, row) in enumerate(zip(vals, ROWS)):
    y = len(ROWS) - 1 - i
    ax.plot([v.normal_mape, v.mape], [y, y], color=MUTED, lw=1, zorder=1)
    ax.scatter(v.normal_mape, y, s=70, color=ORANGE, zorder=2, edgecolor="#fcfcfb", linewidth=1.5, label="Normal 2016~2023" if i == 0 else None)
    ax.scatter(v.mape, y, s=70, color=BLUE, zorder=2, edgecolor="#fcfcfb", linewidth=1.5, label="Rebound 2024~2026.07" if i == 0 else None)
    left, right = (v.mape, v.normal_mape) if v.mape < v.normal_mape else (v.normal_mape, v.mape)
    ax.annotate(f"{left:.2f}", (left, y), xytext=(-9, 0), textcoords="offset points", ha="right", va="center", fontsize=8.5)
    ax.annotate(f"{right:.2f}", (right, y), xytext=(9, 0), textcoords="offset points", ha="left", va="center", fontsize=8.5)
ax.set_yticks(range(len(ROWS)), [r[3] for r in ROWS][::-1], color=INK)
ax.set_xlim(2.5, 9.5)
ax.set_xlabel("Mean MAPE of 1-year-ahead forecasts (%)")
ax.grid(axis="y", visible=False)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
ax.set_title("d=1 wins the normal period; d=0 with exogenous variables wins the rebound", loc="left", fontsize=12)
fig.savefig(FIG / "03_normal_vs_rebound.png")

# --- newlywed coefficient by estimation end: how stable is the level relation?
R = pd.read_csv(REP / "02_normal_by_origin.csv").dropna(subset=["b_newlywed"])
C = pd.read_csv(REP / "03_coefficients.csv")
fig, ax = plt.subplots(figsize=(9, 3.8))
for name, color, text in ((D1, BLUE, "d=1 main order"), (D0, ORANGE, "d=0 alternative")):
    model, track, _ = name.split("|")
    g = R[(R.model == model) & (R.track == track)].sort_values("target")
    c = C[C.name == name].iloc[0]
    x = list(g.target - 1) + [2023]
    b = pd.Series(list(g.b_newlywed) + [c.b_newlywed])
    se = pd.Series(list(g.se_newlywed) + [c.se_newlywed])
    ax.fill_between(x, b - 1.96 * se, b + 1.96 * se, color=color, alpha=0.15, lw=0)
    ax.plot(x, b, color=color, marker="o", ms=5, label=text)
    ax.annotate(text, (2023, b.iloc[-1]), xytext=(8, 0), textcoords="offset points", fontsize=9, va="center")
ax.axhline(0, color=INK, lw=0.8)
ax.set_xlabel("Training data ends in December of")
ax.set_ylabel("Newlywed coefficient (elasticity)")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
ax.set_title("Under d=0 the newlywed coefficient climbs from -0.3 to +0.9; under d=1 it is never significant", loc="left", fontsize=12)
ax.text(2015, 1.42, "band: 95% CI", fontsize=8.5, color=INK)
fig.savefig(FIG / "03_newlywed_coefficient.png")
