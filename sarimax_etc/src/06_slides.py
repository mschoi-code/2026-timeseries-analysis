"""Step 6, SARIMA/SARIMAX section for the intermediate presentation, on the team's slide template.

Usage: python -I src/06_slides.py <team_template.pptx> <output.pptx>   (needs python-pptx, not in the school env)
Numbers come from reports/*.csv so the slides cannot drift from the analysis. Charts stay native so teammates can edit them.
"""
import re
import sys
from copy import deepcopy
from pathlib import Path
from xml.sax.saxutils import escape

import pandas as pd
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION, XL_LEGEND_POSITION, XL_MARKER_STYLE, XL_TICK_LABEL_POSITION, XL_TICK_MARK
from pptx.oxml import parse_xml
from pptx.oxml.ns import nsdecls, qn
from pptx.util import Emu, Inches, Pt

TEMPLATE, OUT = Path(sys.argv[1]), Path(sys.argv[2])
REP = Path(__file__).resolve().parents[1] / "reports"
NAVY, BODY, MUTED, GRID, RULE, GREY = "1F3A5F", "1F2937", "4B5563", "EEF0F3", "D5DBE3", "A6A6A6"
BLUE, ORANGE, GREEN = "2A78D6", "EB6834", "1BAF7A"
FONT = '<a:latin typeface="Pretendard"/><a:ea typeface="Pretendard"/><a:cs typeface="Pretendard"/>'
SUB = lambda s: s.replace("12", "₁₂")

# --- numbers
S = pd.read_csv(REP / "03_rebound_summary.csv")
Q = pd.read_csv(REP / "03_rq2.csv")
D = pd.read_csv(REP / "01_diagnostics.csv")
T = pd.read_csv(REP / "01_differencing_tests.csv").set_index("series")
A = pd.read_csv(REP / "05_outlook_annual.csv")
B = pd.read_csv(REP / "05_backtest_annual.csv")
O = pd.read_csv(REP / "03_rebound_oneshot.csv")
R = pd.read_csv(REP / "02_normal_by_origin.csv")
C = pd.read_csv(REP / "03_coefficients.csv").set_index("name")
M1, M0, MP = "(2,1,2)(0,1,1)12", "(1,0,0)(0,1,1)12+c", "(0,1,1)(0,1,1)12"


def row(model, track, kind):
    return S[(S.model == model) & (S.track == track) & (S.kind == kind)].iloc[0]


naive = S[S.kind == "baseline"].iloc[0]
main, alt = row(M1, "weekdays", "SARIMAX"), row(M0, "weekdays", "SARIMAX")
is_d1 = S.model.str[3] == "1"
is_d0 = S.model.str[3] == "0"
x_d1, x_d0 = S[is_d1 & (S.kind == "SARIMAX")], S[is_d0 & (S.kind == "SARIMAX")]
gain_d1 = Q[Q.key.str[3] == "1"].gain
gain_d0 = Q[Q.key.str[3] == "0"].gain
N = pd.read_csv(REP / "02_normal_summary.csv")
rho_normal, rho_rebound = N.mape.corr(N.rmse, method="spearman"), S.mape.corr(S.rmse, method="spearman")
print(f"RMSE vs MAPE rank correlation: normal {rho_normal:.3f}, rebound {rho_rebound:.3f}")


def signed(v, digits=2):
    return f"{v:+.{digits}f}".replace("-", "−")


def p_txt(p):
    return "<.001" if p < 0.001 else f"{p:.3f}".lstrip("0")


# --- template pieces: reuse the team's exact card and slide-number shapes
prs = Presentation(str(TEMPLATE))
donor = prs.slides[12]
LAYOUT = donor.slide_layout
CARD = deepcopy(next(s for s in donor.shapes if s.name == "Card")._element)
NUM = deepcopy(next(s for s in donor.shapes if s.name.startswith("Slide Number"))._element)
sld_ids = prs.slides._sldIdLst
for sid in list(sld_ids):
    prs.part.drop_rel(sid.get(qn("r:id")))
    sld_ids.remove(sid)
for ext in prs.part._element.xpath(".//p:extLst/p:ext"):  # section lists would point at deleted slides
    if ext.xpath(".//*[local-name()='sectionLst']"):
        ext.getparent().remove(ext)


def next_id(slide):
    return max(int(i) for i in slide.shapes._spTree.xpath(".//p:cNvPr/@id")) + 1


def runs_xml(s, size, color=BODY, b=False):
    """Markup: **navy bold lead-in**, ==yellow highlight==, same as the team's slides."""
    out = ""
    for tok in re.split(r"(\*\*.*?\*\*|==.*?==)", s):
        if not tok:
            continue
        bold, hl, col, text = b, "", color, tok
        if tok.startswith("**"):
            bold, col, text = True, NAVY, tok[2:-2]
        elif tok.startswith("=="):
            hl, text = '<a:highlight><a:srgbClr val="FFFF00"/></a:highlight>', tok[2:-2]
        out += (f'<a:r><a:rPr lang="en-US" sz="{int(size * 100)}" b="{int(bold)}" dirty="0"><a:solidFill><a:srgbClr val="{col}"/></a:solidFill>'
                f'{hl}{FONT}</a:rPr><a:t>{escape(text)}</a:t></a:r>')
    return out


def para_xml(s, size, bullet=False, color=BODY, b=False, algn="l", after=6):
    if bullet:
        ppr = (f'<a:pPr marL="228600" indent="-228600" algn="{algn}" eaLnBrk="0"><a:spcAft><a:spcPts val="{after * 100}"/></a:spcAft>'
               f'<a:buClr><a:srgbClr val="{NAVY}"/></a:buClr><a:buSzPct val="100000"/><a:buChar char="•"/></a:pPr>')
    else:
        ppr = f'<a:pPr marL="0" indent="0" algn="{algn}" eaLnBrk="0"><a:spcAft><a:spcPts val="{after * 100}"/></a:spcAft><a:buNone/></a:pPr>'
    return f"<a:p {nsdecls('a')}>{ppr}{runs_xml(s, size, color, b)}</a:p>"


def new_slide(title):
    s = prs.slides.add_slide(LAYOUT)
    ph = s.placeholders[100]  # the team's title placeholder uses idx 100, not 0
    ph.left, ph.top, ph.width, ph.height = Emu(548640), Emu(320040), Emu(11094415), Emu(731520)  # same box as the team's slides
    tb = ph.text_frame._txBody
    for k, v in dict(wrap="square", lIns="0", tIns="0", rIns="0", bIns="0", anchor="ctr").items():
        tb.find(qn("a:bodyPr")).set(k, v)
    for p in tb.findall(qn("a:p")):
        tb.remove(p)
    tb.append(parse_xml(f"<a:p {nsdecls('a')}><a:pPr marL=\"0\" indent=\"0\" algn=\"l\" eaLnBrk=\"0\"><a:buNone/></a:pPr>{runs_xml(title, 26, NAVY, True)}</a:p>"))
    return s


def finish(s):
    el = deepcopy(NUM)
    el.xpath(".//p:cNvPr")[0].set("id", str(next_id(s)))
    s.shapes._spTree.append(el)


def card(s, x, y, w, h):
    el = deepcopy(CARD)
    el.xpath(".//p:cNvPr")[0].set("id", str(next_id(s)))
    off, ext = el.xpath(".//a:off")[0], el.xpath(".//a:ext[@cx]")[0]
    off.set("x", str(Inches(x))), off.set("y", str(Inches(y)))
    ext.set("cx", str(Inches(w))), ext.set("cy", str(Inches(h)))
    s.shapes._spTree.append(el)


def text(s, x, y, w, h, paras, size=13.5, name=None, anchor="t"):
    """paras: str or (str, options); a str starting with '- ' is a bullet."""
    box = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    if name:
        box.name = name
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    body = tf._txBody
    body.find(qn("a:bodyPr")).set("anchor", anchor)
    for p in body.findall(qn("a:p")):
        body.remove(p)
    for item in paras:
        t, opt = (item, {}) if isinstance(item, str) else item
        bullet = t.startswith("- ")
        body.append(parse_xml(para_xml(t[2:] if bullet else t, opt.pop("size", size), bullet=bullet, **opt)))
    return box


def table(s, x, y, widths, rows, size=12, row_h=0.36, aligns=None, name="Table"):
    frame = s.shapes.add_table(len(rows), len(widths), Inches(x), Inches(y), Inches(sum(widths)), Inches(row_h * len(rows)))
    frame.name = name
    tbl = frame.table
    tbl._tbl.tblPr.getparent().replace(tbl._tbl.tblPr, parse_xml(f"<a:tblPr {nsdecls('a')}/>"))  # the team's tables carry no table style
    for j, w in enumerate(widths):
        tbl.columns[j].width = Inches(w)
    for i, r in enumerate(rows):
        tbl.rows[i].height = Inches(row_h)
        for j, v in enumerate(r):
            tc = tbl.cell(i, j)._tc
            head = i == 0
            algn = (aligns or ["l"] * len(widths))[j]
            body = tc.find(qn("a:txBody"))
            for p in body.findall(qn("a:p")):
                body.remove(p)
            body.append(parse_xml(para_xml(str(v), size, color="FFFFFF" if head else BODY, b=head, algn=algn, after=0)))
            old = tc.find(qn("a:tcPr"))
            if old is not None:
                tc.remove(old)
            none = lambda tag: f'<a:{tag} w="0"><a:noFill/></a:{tag}>'
            bottom = f'<a:lnB w="9525"><a:solidFill><a:srgbClr val="{RULE}"/></a:solidFill></a:lnB>'
            fill = NAVY if head else "FFFFFF"
            tc.append(parse_xml(f'<a:tcPr {nsdecls("a")} marL="73152" marR="73152" marT="27432" marB="27432" anchor="ctr">{none("lnL")}{none("lnR")}{none("lnT")}{bottom}<a:solidFill><a:srgbClr val="{fill}"/></a:solidFill></a:tcPr>'))
    return frame


def style_chart(chart, legend=True, size=11):
    chart.font.size = Pt(size)
    chart.font.name = "Pretendard"
    chart.font.color.rgb = RGBColor.from_string(MUTED)
    chart.has_title = False
    chart.has_legend = legend
    if legend:
        chart.legend.position = XL_LEGEND_POSITION.BOTTOM
        chart.legend.include_in_layout = False


def line_chart(s, x, y, w, h, cats, series, vmin, vmax, major, skip=1, markers=False, fmt="General", alt=""):
    data = CategoryChartData()
    data.categories = [str(c) for c in cats]
    for name, values, *_ in series:
        data.add_series(name, values)
    kind = XL_CHART_TYPE.LINE_MARKERS if markers else XL_CHART_TYPE.LINE
    gf = s.shapes.add_chart(kind, Inches(x), Inches(y), Inches(w), Inches(h), data)
    gf._element.xpath(".//p:cNvPr")[0].set("descr", alt)
    ch = gf.chart
    style_chart(ch)
    va, ca = ch.value_axis, ch.category_axis
    va.minimum_scale, va.maximum_scale, va.major_unit = vmin, vmax, major
    va.has_major_gridlines = True
    va.major_gridlines.format.line.color.rgb = RGBColor.from_string(GRID)
    va.format.line.fill.background()
    va.major_tick_mark = XL_TICK_MARK.NONE
    va.tick_labels.number_format, va.tick_labels.number_format_is_linked = fmt, False
    ca.tick_label_position = XL_TICK_LABEL_POSITION.LOW
    ca.major_tick_mark = XL_TICK_MARK.NONE
    ca.format.line.color.rgb = RGBColor.from_string(MUTED)
    ca.has_major_gridlines = False
    if skip > 1:
        no_multi = ca._element.find(qn("c:noMultiLvlLbl"))
        no_multi.addprevious(parse_xml(f'<c:tickLblSkip {nsdecls("c")} val="{skip}"/>'))
        no_multi.addprevious(parse_xml(f'<c:tickMarkSkip {nsdecls("c")} val="{skip}"/>'))
    for ser, (_, _, color, width) in zip(ch.plots[0].series, series):
        ser.smooth = False
        ser.format.line.color.rgb = RGBColor.from_string(color)
        ser.format.line.width = Pt(width)
        if markers:
            ser.marker.style = XL_MARKER_STYLE.CIRCLE
            ser.marker.size = 8
            ser.marker.format.fill.solid()
            ser.marker.format.fill.fore_color.rgb = RGBColor.from_string(color)
            ser.marker.format.line.color.rgb = RGBColor.from_string("FFFFFF")
        else:
            ser.marker.style = XL_MARKER_STYLE.NONE
    return ch


# --- 1. procedure
s = new_slide("SARIMA·SARIMAX: Selection Separated from Evaluation")
stages = [("Stage 1", "Box-Jenkins", "Data through 2015", ["- ADF, KPSS, ACF·PACF for d and D", "- AIC, compared within the same d", "- Ljung-Box at lags 12 and 24"]),
          ("Stage 2", "Normal period", "2016~2023", ["- Eight December origins, expanding window", "- Coefficients re-estimated at each origin", "- ==Locks the main model=="]),
          ("Stage 3", "Rebound period", "2024.01~2026.07", ["- Coefficients fixed in Dec 2023", "- 20 origins, 1-year-ahead paths", "- ==Reports only, changes nothing=="])]
for i, (stage, head, span, items) in enumerate(stages):
    x = 0.6 + i * (3.85 + 0.29)
    card(s, x, 1.3, 3.85, 2.95)
    text(s, x + 0.25, 1.47, 3.35, 2.7, [(stage, dict(size=12, color=MUTED, after=2)), (head, dict(size=18, color=NAVY, b=True, after=2)), (span, dict(size=13.5, color=NAVY, after=10)), *items], size=14, name=f"{stage} text")
card(s, 0.6, 4.5, 12.13, 1.6)
text(s, 0.85, 4.7, 11.63, 1.6, [
    "- **Candidates:** the Box-Jenkins pick for each d, the proposal's order and an AR(1) order, each with and without a weekday regressor",
    "- **Main-model rule:** ==lowest normal-period MAPE among SARIMAX models that pass Ljung-Box through 2015==; within 0.1 points, lower RMSE",
    "- **Metric:** mean MAPE of 1-year-ahead forecasts, with RMSE as a secondary check that the rankings agree",
], size=14, name="Rules")
finish(s)

# --- 2. order selection
s = new_slide("Order Selection: Residuals Reveal a Calendar Effect")
card(s, 0.6, 1.3, 5.05, 2.6)
text(s, 0.85, 1.45, 4.6, 0.35, [("Differencing, data through 2015", dict(size=14, color=NAVY, b=True))], name="Differencing heading")
names = {"log y": "log y", "D1": "Δ₁", "D12": "Δ₁₂", "D1 D12": "Δ₁Δ₁₂"}
fmt_p = lambda p: "<.01" if p < 0.01 else (">.10" if p >= 0.1 else f"{p:.3f}".lstrip("0"))
rows = [["Series", "ADF p", "KPSS p", "ACF lag 12"]] + [[names[k], fmt_p(T.adf_c[k]), fmt_p(T.kpss_c[k]), f"{T.acf12[k]:.2f}".replace("-", "−")] for k in names]
table(s, 0.85, 1.85, [1.3, 1.0, 1.0, 1.25], rows, aligns=["l", "r", "r", "r"], name="Differencing tests")
card(s, 5.85, 1.3, 6.88, 2.6)
text(s, 6.1, 1.45, 6.4, 0.35, [("Ljung-Box p at lags 12, 24 (SARIMA, through 2015)", dict(size=14, color=NAVY, b=True))], name="Ljung-Box heading")
rows = [["Order", "No weekdays", "+ Weekdays", "AIC change"]]
for model in (MP, M1, M0):
    a, b = D[(D.model == model) & (D.track == "proposal")].iloc[0], D[(D.model == model) & (D.track == "weekdays")].iloc[0]
    rows.append([SUB(model), f"{p_txt(a.sarima_lb12)}, {p_txt(a.sarima_lb24)}", f"{p_txt(b.sarima_lb12)}, {p_txt(b.sarima_lb24)}", f"{b.sarima_aic - a.sarima_aic:+.1f}".replace("-", "−")])
table(s, 6.1, 1.85, [2.45, 1.4, 1.3, 1.2], rows, row_h=0.45, aligns=["l", "r", "r", "r"], name="Ljung-Box table")
text(s, 0.6, 4.15, 12.13, 2.7, [
    f"- **D=1:** the ACF of Δ₁ stays high at seasonal lags ({T.acf12['D1']:.2f}, {T.acf24['D1']:.2f}, {T.acf36['D1']:.2f} at lags 12, 24, 36)",
    "- **d:** on Δ₁₂ the ADF and KPSS tests disagree; on Δ₁Δ₁₂ both indicate stationarity. ==We keep d=1 and d=0 as candidates==",
    "- **Orders:** AIC picks (2,1,2)(0,1,1)₁₂ for d=1. Without weekdays, ==all 36 d=0 orders and the proposal's order fail Ljung-Box==",
    "- **Weekdays:** log weekdays per month clears the residual autocorrelation (coefficient +0.22, se 0.04). Also removing Seollal and Chuseok did not improve on it",
], size=14.5, name="Findings")
finish(s)

# --- 3. RQ1
s = new_slide("RQ1: d=1 Wins the Normal Period, but Not the Rebound")
card(s, 0.6, 1.3, 8.35, 5.4)
text(s, 0.85, 1.42, 7.9, 0.32, [("Mean MAPE of 1-year-ahead forecasts (%)", dict(size=13, color=NAVY, b=True))], name="Chart heading")
bars = [("Seasonal naive", naive), ("Proposal order, SARIMA", row(MP, "proposal", "SARIMA")), ("Proposal order, SARIMAX", row(MP, "proposal", "SARIMAX")),
        ("d=1 main order, SARIMA", row(M1, "weekdays", "SARIMA")), ("d=1 main order, SARIMAX", main), ("d=0 alternative, SARIMA", row(M0, "weekdays", "SARIMA")), ("d=0 alternative, SARIMAX", alt)]
data = CategoryChartData()
data.categories = [b[0] for b in bars]
data.add_series("Normal 2016~2023", [round(b[1].normal_mape, 2) for b in bars])
data.add_series("Rebound 2024~2026.07", [round(b[1].mape, 2) for b in bars])
gf = s.shapes.add_chart(XL_CHART_TYPE.BAR_CLUSTERED, Inches(0.75), Inches(1.72), Inches(8.05), Inches(4.9), data)
gf._element.xpath(".//p:cNvPr")[0].set("descr", "Clustered bars of normal-period and rebound-period MAPE for seven models; the table in reports/03_rebound_summary.csv has every value.")
ch = gf.chart
style_chart(ch)
pl = ch.plots[0]
pl.gap_width, pl.overlap = 45, -5
pl.has_data_labels = True
pl.data_labels.number_format, pl.data_labels.number_format_is_linked = "0.00", False
pl.data_labels.position = XL_LABEL_POSITION.OUTSIDE_END
pl.data_labels.show_value = True
pl.data_labels.font.size = Pt(11)
pl.data_labels.font.color.rgb = RGBColor.from_string(BODY)
for ser, color in zip(pl.series, (GREY, NAVY)):
    ser.format.fill.solid()
    ser.format.fill.fore_color.rgb = RGBColor.from_string(color)
ch.category_axis.reverse_order = True
ch.category_axis.major_tick_mark = XL_TICK_MARK.NONE
ch.category_axis.format.line.color.rgb = RGBColor.from_string(MUTED)
ch.category_axis.tick_labels.font.color.rgb = RGBColor.from_string(BODY)
ch.value_axis.visible = False
ch.value_axis.has_major_gridlines = False
ch.value_axis.minimum_scale, ch.value_axis.maximum_scale = 0, 10
card(s, 9.15, 1.3, 3.58, 5.4)
text(s, 9.38, 1.5, 3.12, 5.05, [
    (f"{main.normal_mape:.2f}%", dict(size=30, color=NAVY, b=True, after=0)), ("Main model, normal period", dict(size=12, color=MUTED, after=8)),
    (f"{main.mape:.2f}%", dict(size=30, color=NAVY, b=True, after=0)), (f"Main model, rebound; seasonal naive is {naive.mape:.2f}%", dict(size=12, color=MUTED, after=10)),
    (f"**Main model:** {SUB(M1)} + weekdays + newlywed + women", dict(size=12.5, after=8)),
    (f"- ==All models under-forecast== (mean bias {signed(S.bias.max(), 1)}% to {signed(S.bias.min(), 1)}%)", dict(size=12.5, after=6)),
    (f"- At 12 months ahead, naive ({naive.h12_mape:.2f}%) beats every d=1 model ({S[is_d1].h12_mape.min():.2f}~{S[is_d1].h12_mape.max():.2f}%)", dict(size=12.5, after=6)),
    (f"- RMSE rankings agree with MAPE (rank correlation {rho_normal:.2f} normal, {rho_rebound:.2f} rebound)", dict(size=12.5, after=0)),
], name="Key numbers")
finish(s)

# --- 4. RQ2
s = new_slide("RQ2: Exogenous Variables Help Clearly Only Under d=0")
card(s, 0.6, 1.3, 6.95, 3.95)
text(s, 0.85, 1.4, 6.5, 0.32, [("One forecast path from Dec 2023, monthly births (thousands)", dict(size=13, color=NAVY, b=True))], name="Chart heading")
births = pd.read_csv(Path(__file__).resolve().parents[1] / "data" / "raw" / "월간_인구동향_출생_혼인.csv", skiprows=2, header=None, names=["t", "b", "m"], dtype={"t": str})
births.index = births.t.str.slice(0, 7).str.replace(".", "-", regex=False)
births = births.loc["2022-01":"2026-07"]
months = list(births.index)
path = lambda name: [None if pd.isna(v) else round(float(v), 2) for v in (O[O.name == name].assign(k=lambda f: f.target.str.slice(0, 7)).set_index("k").forecast / 1000).reindex(months)]
line_chart(s, 0.72, 1.7, 6.72, 3.5, months, [("Actual", (births.b / 1000).round(2).tolist(), NAVY, 2.25), ("d=1 main SARIMAX", path(f"{M1}|weekdays|SARIMAX"), BLUE, 2.0), ("d=0 alternative SARIMAX", path(f"{M0}|weekdays|SARIMAX"), ORANGE, 2.0)],
           14, 28, 2, skip=6, alt="Line chart of actual monthly births and two forecast paths from December 2023: the d=1 path keeps falling and the d=0 path turns upward; both stay below actual births.")
card(s, 7.75, 1.3, 4.98, 3.95)
text(s, 7.97, 1.4, 4.6, 0.32, [("Rebound MAPE (%), same order and calendar", dict(size=13, color=NAVY, b=True))], name="Table heading")
rows = [["Order", "SARIMA", "SARIMAX", "Gain"]]
order = [f"{MP}|proposal", f"{MP}|weekdays", f"{M1}|proposal", f"{M1}|weekdays", f"{M0}|proposal", f"{M0}|weekdays", "(2,0,0)(0,1,1)12+c|proposal"]
for key in order:
    q = Q[Q.key == key].iloc[0]
    m, track = key.split("|")
    g = signed(q.gain)
    rows.append([SUB(m) + (" + wd" if track == "weekdays" else ""), f"{q.sarima:.2f}", f"{q.sarimax:.2f}", f"=={g}==" if m[3] == "0" else g])
table(s, 7.97, 1.76, [2.35, 0.75, 0.82, 0.62], rows, size=11, row_h=0.4, aligns=["l", "r", "r", "r"], name="RQ2 table")
text(s, 0.6, 5.45, 12.13, 1.45, [
    f"- **Result:** adding newlywed couples and women 30~34 changes rebound MAPE by {signed(gain_d1.min())}~{signed(gain_d1.max())} points under d=1 and ==by {signed(gain_d0.min())}~{signed(gain_d0.max())} points under d=0==",
    "- **Why they split:** the newlywed signal is a slow level change. d=1 differences it away, while d=0 keeps the level link and turns upward with marriages",
    "- **Reading the table:** wd = weekday regressor; gain = SARIMA minus SARIMAX, positive is better",
], name="Findings")
finish(s)

# --- 5. why d=1 stays
s = new_slide("Why d=1 Stays the Main Model")
card(s, 0.6, 1.3, 6.95, 5.4)
text(s, 0.85, 1.42, 6.5, 0.32, [("Newlywed coefficient by training end (elasticity)", dict(size=13, color=NAVY, b=True))], name="Chart heading")
years = list(range(2015, 2024))
coef = {}
for key, model in (("d1", M1), ("d0", M0)):
    g = R[(R.model == model) & (R.track == "weekdays") & (R.kind == "SARIMAX")].sort_values("target")
    c = C.loc[f"{model}|weekdays|SARIMAX"]
    coef[key] = (pd.Series(list(g.b_newlywed) + [c.b_newlywed], index=years), pd.Series(list(g.se_newlywed) + [c.se_newlywed], index=years))
sig = lambda k: (coef[k][0] - 1.96 * coef[k][1] > 0) | (coef[k][0] + 1.96 * coef[k][1] < 0)
assert not sig("d1").any(), "slide says the d=1 coefficient is never significant"
first_sig = int(sig("d0")[sig("d0")].index[0])
assert sig("d0").loc[first_sig:].all(), "slide says the d=0 coefficient stays significant from then on"
line_chart(s, 0.72, 1.72, 6.72, 4.3, years, [("d=1 main order", coef["d1"][0].round(2).tolist(), BLUE, 2.0), ("d=0 alternative", coef["d0"][0].round(2).tolist(), ORANGE, 2.0)], -1.2, 1.2, 0.4, markers=True, fmt="0.0",
           alt="Line chart of the newlywed coefficient by last training year: under d=0 it climbs from about -0.3 to about 0.9; under d=1 it stays near or below zero.")
text(s, 0.85, 6.05, 6.5, 0.55, [(f"95% CI excludes zero for d=0 from {first_sig} on, and never for d=1", dict(size=12, color=MUTED))], name="Chart note")
wy16 = R[(R.model == M0) & (R.track == "weekdays") & (R.kind == "SARIMAX") & (R.target == 2016)].mape.iloc[0]
card(s, 7.75, 1.3, 4.98, 2.95)
text(s, 7.97, 1.45, 4.55, 2.7, [("Against d=0", dict(size=14, color=NAVY, b=True, after=6)),
    (f"- Normal-period MAPE =={x_d0.normal_mape.min():.2f}~{x_d0.normal_mape.max():.2f}%== vs {x_d1.normal_mape.min():.2f}~{x_d1.normal_mape.max():.2f}% for d=1 ({wy16:.1f}% in 2016)", dict(size=14)),
    ("- No level link through 2015 (Engle-Granger p = .638)", dict(size=14)),
    (f"- Coefficient moves from {signed(coef['d0'][0][2015])} to {signed(coef['d0'][0][2023])}", dict(size=14)),
    ("- The rebound has only one turning point", dict(size=14, after=0))], name="Against d=0")
card(s, 7.75, 4.45, 4.98, 2.25)
text(s, 7.97, 4.58, 4.55, 2.05, [("For d=0", dict(size=14, color=NAVY, b=True, after=6)),
    (f"- Rebound MAPE {alt.mape:.2f}%, the lowest of all models", dict(size=14)),
    ("- With weekdays, its residuals pass Ljung-Box", dict(size=14)),
    ("- The 12~35-month window beat later windows (7.1% vs 9.7~19.9%, one path)", dict(size=14, after=0))], name="For d=0")
assert alt.mape == S.mape.min()
finish(s)

# --- 6. RQ3
s = new_slide("RQ3: Rebound Holds to 2027, Then Depends on Marriages and d")
card(s, 0.6, 1.3, 12.13, 3.7)
act = pd.read_csv(Path(__file__).resolve().parents[1] / "data" / "raw" / "월간_인구동향_출생_혼인.csv", skiprows=2, header=None, names=["t", "b", "m"], dtype={"t": str})
act = act[~act.t.str.startswith("2026")].assign(y=lambda f: f.t.str.slice(0, 4).astype(int)).groupby("y").b.sum() / 1000
cats = list(range(2015, 2032))
for i, (mname, head) in enumerate((("d=1 main", "d=1 main model"), ("d=0 alternative", "d=0 alternative"))):
    x = 0.8 + i * 6.0
    text(s, x + 0.1, 1.38, 5.6, 0.3, [(f"{head}: annual births (thousands)", dict(size=13, color=NAVY, b=True))], name=f"{head} heading")
    g = A[A.model == mname]
    series = [("Actual", [round(float(act[y]), 1) if y <= 2025 else None for y in cats], NAVY, 2.25)]
    for sc, color in (("High", GREEN), ("Hold", BLUE), ("Low", ORANGE)):
        v = g[g.scenario == sc].set_index("year").births / 1000
        series.append((sc, [round(float(act[2025]), 1) if y == 2025 else (round(float(v[y]), 1) if y > 2025 else None) for y in cats], color, 2.0))
    line_chart(s, x, 1.66, 5.75, 3.3, cats, series, 200, 450, 50, skip=4, alt=f"{head}: actual annual births fall to 2023 and rise to 2025; three marriage scenarios run from 2026 to 2031.")
hold = A[A.scenario == "Hold"]
y27, y31 = A[A.year == 2027].births / 1000, A[A.year == 2031]
b5 = B[B.ahead == 5].assign(a=lambda f: f.error.abs()).groupby("name").a.mean()
b1 = B[B.ahead == 1].assign(a=lambda f: f.error.abs()).groupby("name").a.mean()
rng = lambda m: f"{y31[y31.model == m].births.min() / 1000:.0f}~{y31[y31.model == m].births.max() / 1000:.0f}k"
assert (A[A.year == 2026].births / 1000).round(0).nunique() == 1 and (A[A.year == 2027].births.values > A[A.year == 2026].births.values).all()
text(s, 0.6, 5.17, 12.13, 1.75, [
    "- **Scenarios:** marriages +5%/year (High, 2026 pace), flat (Hold), −5%/year (Low, women 25~29 shrink 5.1%/year). Refit on data through Jul 2026",
    f"- **Through 2027:** every path rises, to {A[A.year == 2026].births.iloc[0] / 1000:.0f}k in 2026 and ==~{y27.min():.0f}~{y27.max():.0f}k in 2027==; marriages already fix newlywed couples through Jul 2027".replace("==~", "=="),
    f"- **By 2031:** {rng('d=1 main')} under d=1 and {rng('d=0 alternative')} under d=0. Under Hold, d=0 peaks in 2028 while d=1 still rises in 2031",
    f"- **Trust:** backtest annual error grows from {b1['d=1 main SARIMAX']:.1f}% at 1 year to =={b5['d=1 main SARIMAX']:.0f}% at 5 years for d=1==, and from {b1['d=0 alternative SARIMAX']:.1f}% to =={b5['d=0 alternative SARIMAX']:.0f}% for d=0==. Read 2029~2031 as scenarios, not forecasts",
], size=13, name="Findings")
peak = {k: int(g.loc[g.births.idxmax(), "year"]) for k, g in hold.groupby("model")}
assert peak == {"d=0 alternative": 2028, "d=1 main": 2031}, peak
finish(s)

# --- 7. limitations and next steps
s = new_slide("Limitations and Next Steps")
card(s, 0.6, 1.3, 5.95, 3.95)
text(s, 0.85, 1.5, 5.45, 4.4, [("Limitations", dict(size=18, color=NAVY, b=True, after=10)),
    "- **One event:** the rebound result rests on a single turning point",
    "- **Not fully blind:** some rebound errors were seen before the rules were written",
    "- **Added variable:** the weekday regressor is not in the proposal",
    "- **Look-ahead:** women 30~34 comes from the Dec 2023 projection, which is future information for the normal period",
    "- **Data:** 2026 values are preliminary and end in July"], size=15, name="Limitations")
card(s, 6.78, 1.3, 5.95, 3.95)
text(s, 7.03, 1.5, 5.45, 4.4, [("Next Steps", dict(size=18, color=NAVY, b=True, after=10)),
    "- **Compare:** ==regression, Holt-Winters and VAR on the same metric==",
    "- **Update:** add Aug 2026 and later months to the outlook",
    "- **Level link:** test its stability formally, since RQ2 depends on it",
    "- **Variables:** explore youth employment and housing prices"], size=15, name="Next steps")
finish(s)

prs.core_properties.title = "Team 21 Intermediate Presentation: SARIMA and SARIMAX"
prs.save(str(OUT))
print("saved", OUT, "slides", len(prs.slides))
