"""
briefing.py — one-page commander briefing PDF for a predicted incident.

Turns a GridlockPredictor output into a printable operational briefing a control-room
officer could act on: severity, recommended response, the reasoning, and historical context.
Returns PDF bytes (for a Streamlit download button) — no file written to disk.
"""

from io import BytesIO
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle)
from reportlab.lib.enums import TA_LEFT

INK = colors.HexColor("#0E1726")
AMBER = colors.HexColor("#C77800")
TEAL = colors.HexColor("#0F766E")
RED = colors.HexColor("#B91C1C")
GREEN = colors.HexColor("#15803D")
SLATE = colors.HexColor("#475569")
LINE = colors.HexColor("#CBD5E1")
PANEL = colors.HexColor("#F1F5F9")

TIER_COLOR = {"CRITICAL": RED, "HIGH": AMBER, "MODERATE": TEAL, "LOW": GREEN}


def _styles():
    s = getSampleStyleSheet()
    s.add(ParagraphStyle("Brand", fontName="Helvetica-Bold", fontSize=16, textColor=INK, leading=18))
    s.add(ParagraphStyle("Sub", fontName="Helvetica", fontSize=8.5, textColor=SLATE, leading=11))
    s.add(ParagraphStyle("H", fontName="Helvetica-Bold", fontSize=10.5, textColor=INK, leading=14,
                         spaceBefore=6, spaceAfter=3))
    s.add(ParagraphStyle("Body", fontName="Helvetica", fontSize=9.5, textColor=INK, leading=13))
    s.add(ParagraphStyle("Small", fontName="Helvetica", fontSize=8.5, textColor=SLATE, leading=11))
    s.add(ParagraphStyle("Cell", fontName="Helvetica", fontSize=9, textColor=INK, leading=12))
    s.add(ParagraphStyle("CellB", fontName="Helvetica-Bold", fontSize=9, textColor=INK, leading=12))
    return s


def build_briefing(event: dict, result: dict) -> bytes:
    """event = the input dict, result = GridlockPredictor.predict() output. Returns PDF bytes."""
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=16 * mm, bottomMargin=14 * mm,
                            leftMargin=16 * mm, rightMargin=16 * mm)
    s = _styles()
    story = []

    # ---- header ----
    ts = datetime.now().strftime("%d %b %Y · %H:%M")
    story.append(Paragraph("GRIDLOCK ORACLE — INCIDENT BRIEFING", s["Brand"]))
    story.append(Paragraph(f"Operational decision support · generated {ts}", s["Sub"]))
    story.append(Spacer(1, 6))
    story.append(Table([[""]], colWidths=[178 * mm], style=TableStyle(
        [("LINEBELOW", (0, 0), (-1, -1), 1, INK)])))
    story.append(Spacer(1, 8))

    # ---- severity banner ----
    tier = str(result.get("impact_tier", "—")).upper()
    tcol = TIER_COLOR.get(tier, SLATE)
    prob = result.get("closure_prob", 0)
    readiness = result.get("readiness_tier", "—")
    banner = Table([[
        Paragraph(f"<b>SEVERITY: {tier}</b>", ParagraphStyle("b1", fontName="Helvetica-Bold",
                  fontSize=13, textColor=colors.white)),
        Paragraph(f"Closure probability<br/><b>{prob*100:.0f}%</b>", ParagraphStyle("b2",
                  fontName="Helvetica", fontSize=9, textColor=colors.white, leading=13)),
        Paragraph(f"Readiness<br/><b>{readiness}</b>", ParagraphStyle("b3",
                  fontName="Helvetica", fontSize=9, textColor=colors.white, leading=13)),
        Paragraph(f"Impact score<br/><b>{result.get('impact_score','—')} / 10</b>", ParagraphStyle("b4",
                  fontName="Helvetica", fontSize=9, textColor=colors.white, leading=13)),
    ]], colWidths=[64 * mm, 38 * mm, 38 * mm, 38 * mm])
    banner.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), tcol), ("TOPPADDING", (0, 0), (-1, -1), 9),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 9), ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
    story.append(banner)
    story.append(Spacer(1, 10))

    # ---- incident details ----
    story.append(Paragraph("INCIDENT", s["H"]))
    det = [
        ["Cause", str(event.get("event_cause", "—")), "Type", str(event.get("event_type", "—"))],
        ["Vehicle", str(event.get("veh_type", "—")), "Priority", str(event.get("priority", "—"))],
        ["Time", str(event.get("start_datetime", "—")), "Police station", str(event.get("police_station", "—"))],
    ]
    t = Table([[Paragraph(c, s["CellB"]) if i % 2 == 0 else Paragraph(c, s["Cell"])
                for i, c in enumerate(row)] for row in det],
              colWidths=[28 * mm, 61 * mm, 32 * mm, 57 * mm])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), PANEL), ("GRID", (0, 0), (-1, -1), 0.5, LINE),
                           ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                           ("LEFTPADDING", (0, 0), (-1, -1), 7)]))
    story.append(t)
    story.append(Spacer(1, 10))

    # ---- recommended response ----
    res = result.get("resources", {}) or {}
    story.append(Paragraph("RECOMMENDED RESPONSE", s["H"]))
    rec = [
        ["Personnel", str(res.get("personnel", "—")), "Barricading", "Yes" if res.get("barricading_recommended") else "No"],
        ["Supervisors", str(res.get("supervisors", "—")), "Diversion", "Yes" if res.get("diversion_recommended") else "No"],
        ["Barricades", str(res.get("barricades", "—")), "Rapid response", "Yes" if res.get("rapid_response_required") else "No"],
        ["Est. clearance", f"{result.get('expected_clearance_mins','—')} min", "Basis", str(res.get("basis", "model + analogs"))[:46]],
    ]
    rt = Table([[Paragraph(c, s["CellB"]) if i % 2 == 0 else Paragraph(c, s["Cell"])
                 for i, c in enumerate(row)] for row in rec],
               colWidths=[30 * mm, 35 * mm, 32 * mm, 81 * mm])
    rt.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, LINE), ("TOPPADDING", (0, 0), (-1, -1), 5),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 5), ("LEFTPADDING", (0, 0), (-1, -1), 7)]))
    story.append(rt)
    story.append(Spacer(1, 10))

    # ---- why this prediction ----
    exps = result.get("explanations", []) or []
    if exps:
        story.append(Paragraph("WHY THIS PREDICTION (top factors)", s["H"]))
        rows = []
        for e in exps[:5]:
            d = e.get("direction", "")
            arrow = "▲ raises" if d == "raises" else "▼ lowers"
            col = RED if d == "raises" else GREEN
            rows.append([Paragraph(e.get("label", e.get("feature", "")), s["Cell"]),
                         Paragraph(f'<font color="{col.hexval()}"><b>{arrow}</b></font>', s["Cell"]),
                         Paragraph(f'{e.get("contribution", 0):+.2f}', s["Cell"])])
        et = Table(rows, colWidths=[95 * mm, 50 * mm, 33 * mm])
        et.setStyle(TableStyle([("LINEBELOW", (0, 0), (-1, -2), 0.4, LINE),
                                ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]))
        story.append(et)
        story.append(Spacer(1, 6))
        story.append(Paragraph("Contributions are exact SHAP values for this specific incident "
                               "(XGBoost pred_contribs).", s["Small"]))
        story.append(Spacer(1, 8))

    # ---- historical context ----
    ana = result.get("analogs")
    if ana:
        story.append(Paragraph("HISTORICAL CONTEXT", s["H"]))
        story.append(Paragraph(
            f"Based on {ana.get('n_matched','—')} most-similar past incidents: typical clearance "
            f"{ana.get('clearance_p25','—')}–{ana.get('clearance_p75','—')} min; "
            f"{ana.get('analog_closure_rate',0)*100:.0f}% required a road closure.", s["Body"]))
        story.append(Spacer(1, 8))

    # ---- footer ----
    story.append(Spacer(1, 6))
    story.append(Table([[""]], colWidths=[178 * mm], style=TableStyle([("LINEABOVE", (0, 0), (-1, -1), 0.5, LINE)])))
    story.append(Paragraph(
        "Decision-support recommendation generated by Gridlock Oracle. Probabilities are calibrated; "
        "resource figures are guidance based on historical analogs. Final deployment is at the "
        "commanding officer's discretion.", s["Small"]))

    doc.build(story)
    buf.seek(0)
    return buf.getvalue()