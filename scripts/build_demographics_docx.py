from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from docx import Document
from docx.enum.section import WD_ORIENT
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


OUTPUT_DIR = Path("outputs/demographics_npj_nature")
ANALYSIS_JSON = OUTPUT_DIR / "demographic_analysis.json"
OUTPUT_DOCX = OUTPUT_DIR / "Table1_demographics_npj_nature.docx"

INK = "000000"
MUTED = "5B6573"
LIGHT = "D9DEE5"
PALE = "F5F7F9"
ACCENT = "1F4D78"
TABLE_WIDTH_DXA = 12960


def set_run_font(run, size=10.5, bold=False, italic=False, color=INK, name="Calibri"):
    run.font.name = name
    run._element.get_or_add_rPr().rFonts.set(qn("w:ascii"), name)
    run._element.get_or_add_rPr().rFonts.set(qn("w:hAnsi"), name)
    run._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
    run.font.size = Pt(size)
    run.bold = bold
    run.italic = italic
    run.font.color.rgb = RGBColor.from_string(color)


def set_cell_shading(cell, fill):
    tc_pr = cell._tc.get_or_add_tcPr()
    shading = tc_pr.find(qn("w:shd"))
    if shading is None:
        shading = OxmlElement("w:shd")
        tc_pr.append(shading)
    shading.set(qn("w:fill"), fill)


def set_cell_margins(cell, top=80, start=120, bottom=80, end=120):
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for margin, value in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = tc_mar.find(qn(f"w:{margin}"))
        if node is None:
            node = OxmlElement(f"w:{margin}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def set_table_geometry(table, widths_dxa, indent_dxa=120):
    table.autofit = False
    table.alignment = WD_TABLE_ALIGNMENT.LEFT
    tbl_pr = table._tbl.tblPr
    tbl_w = tbl_pr.find(qn("w:tblW"))
    if tbl_w is None:
        tbl_w = OxmlElement("w:tblW")
        tbl_pr.append(tbl_w)
    tbl_w.set(qn("w:w"), str(sum(widths_dxa)))
    tbl_w.set(qn("w:type"), "dxa")
    tbl_ind = tbl_pr.find(qn("w:tblInd"))
    if tbl_ind is None:
        tbl_ind = OxmlElement("w:tblInd")
        tbl_pr.append(tbl_ind)
    tbl_ind.set(qn("w:w"), str(indent_dxa))
    tbl_ind.set(qn("w:type"), "dxa")
    layout = tbl_pr.find(qn("w:tblLayout"))
    if layout is None:
        layout = OxmlElement("w:tblLayout")
        tbl_pr.append(layout)
    layout.set(qn("w:type"), "fixed")

    grid = table._tbl.tblGrid
    for child in list(grid):
        grid.remove(child)
    for width in widths_dxa:
        col = OxmlElement("w:gridCol")
        col.set(qn("w:w"), str(width))
        grid.append(col)
    for row in table.rows:
        for index, cell in enumerate(row.cells):
            cell.width = Inches(widths_dxa[index] / 1440)
            tc_pr = cell._tc.get_or_add_tcPr()
            tc_w = tc_pr.find(qn("w:tcW"))
            if tc_w is None:
                tc_w = OxmlElement("w:tcW")
                tc_pr.append(tc_w)
            tc_w.set(qn("w:w"), str(widths_dxa[index]))
            tc_w.set(qn("w:type"), "dxa")
            set_cell_margins(cell)


def set_table_borders(table, header_bottom=True):
    tbl_pr = table._tbl.tblPr
    borders = tbl_pr.find(qn("w:tblBorders"))
    if borders is None:
        borders = OxmlElement("w:tblBorders")
        tbl_pr.append(borders)
    for edge in ("top", "bottom"):
        node = OxmlElement(f"w:{edge}")
        node.set(qn("w:val"), "single")
        node.set(qn("w:sz"), "10" if edge == "top" else "8")
        node.set(qn("w:color"), INK)
        borders.append(node)
    for edge in ("left", "right", "insideH", "insideV"):
        node = OxmlElement(f"w:{edge}")
        node.set(qn("w:val"), "nil")
        borders.append(node)
    if header_bottom:
        tr_pr = table.rows[0]._tr.get_or_add_trPr()
        tr_borders = OxmlElement("w:trBorders")
        bottom = OxmlElement("w:bottom")
        bottom.set(qn("w:val"), "single")
        bottom.set(qn("w:sz"), "8")
        bottom.set(qn("w:color"), INK)
        tr_borders.append(bottom)
        tr_pr.append(tr_borders)


def set_repeat_table_header(row):
    tr_pr = row._tr.get_or_add_trPr()
    header = OxmlElement("w:tblHeader")
    header.set(qn("w:val"), "true")
    tr_pr.append(header)


def set_keep_with_next(paragraph, value=True):
    p_pr = paragraph._p.get_or_add_pPr()
    node = p_pr.find(qn("w:keepNext"))
    if value and node is None:
        node = OxmlElement("w:keepNext")
        p_pr.append(node)
    elif not value and node is not None:
        p_pr.remove(node)


def style_cell(cell, text, *, bold=False, align=WD_ALIGN_PARAGRAPH.LEFT, size=8.1, color=INK, fill=None):
    cell.text = ""
    paragraph = cell.paragraphs[0]
    paragraph.alignment = align
    paragraph.paragraph_format.space_before = Pt(0)
    paragraph.paragraph_format.space_after = Pt(0)
    paragraph.paragraph_format.line_spacing = 1.0
    run = paragraph.add_run(str(text))
    set_run_font(run, size=size, bold=bold, color=color, name="Arial")
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
    if fill:
        set_cell_shading(cell, fill)


def add_heading(doc, text, level=1):
    paragraph = doc.add_paragraph(style=f"Heading {level}")
    paragraph.add_run(text)
    set_keep_with_next(paragraph)
    return paragraph


def add_body(doc, text, *, size=9.5, color=INK, italic=False, after=5):
    paragraph = doc.add_paragraph()
    paragraph.paragraph_format.space_before = Pt(0)
    paragraph.paragraph_format.space_after = Pt(after)
    paragraph.paragraph_format.line_spacing = 1.15
    run = paragraph.add_run(text)
    set_run_font(run, size=size, italic=italic, color=color)
    return paragraph


def add_title_block(doc):
    kicker = doc.add_paragraph()
    kicker.paragraph_format.space_before = Pt(0)
    kicker.paragraph_format.space_after = Pt(3)
    set_run_font(kicker.add_run("RESEARCH TABLE  |  MANUSCRIPT-READY"), size=8.5, bold=True, color=ACCENT, name="Arial")
    title = doc.add_paragraph()
    title.paragraph_format.space_before = Pt(0)
    title.paragraph_format.space_after = Pt(3)
    set_run_font(title.add_run("Demographic and imaging characteristics by MRI device group"), size=20, bold=True, name="Arial")
    subtitle = doc.add_paragraph()
    subtitle.paragraph_format.space_before = Pt(0)
    subtitle.paragraph_format.space_after = Pt(10)
    set_run_font(subtitle.add_run("npj Digital Medicine / Nature Portfolio-style presentation"), size=10.5, italic=True, color=MUTED, name="Arial")

    meta = doc.add_table(rows=1, cols=4)
    values = [
        ("Population", "288 participants"),
        ("Comparison", "SIEMENS vs UIH"),
        ("Unit", "Participant"),
        ("Generated", date.today().strftime("%d %b %Y")),
    ]
    for cell, (label, value) in zip(meta.rows[0].cells, values):
        cell.text = ""
        p = cell.paragraphs[0]
        p.paragraph_format.space_after = Pt(0)
        set_run_font(p.add_run(label.upper() + "\n"), size=7.2, bold=True, color=MUTED, name="Arial")
        set_run_font(p.add_run(value), size=9.2, bold=True, name="Arial")
        set_cell_shading(cell, PALE)
        set_cell_margins(cell, top=90, bottom=90, start=130, end=130)
    set_table_geometry(meta, [3240, 3240, 3240, 3240], indent_dxa=120)
    meta_borders = meta._tbl.tblPr.find(qn("w:tblBorders"))
    if meta_borders is None:
        meta_borders = OxmlElement("w:tblBorders")
        meta._tbl.tblPr.append(meta_borders)
    for edge in ("top", "bottom", "left", "right", "insideH", "insideV"):
        node = OxmlElement(f"w:{edge}")
        node.set(qn("w:val"), "nil")
        meta_borders.append(node)
    doc.add_paragraph().paragraph_format.space_after = Pt(1)


def add_main_table(doc, analysis):
    heading = doc.add_paragraph()
    heading.paragraph_format.space_before = Pt(0)
    heading.paragraph_format.space_after = Pt(5)
    set_run_font(heading.add_run("Table 1. Demographic, clinical, and imaging characteristics by MRI device group"), size=10, bold=True, name="Arial")
    set_keep_with_next(heading)

    headers = ["Characteristic", "Overall\n(N=288)", "SIEMENS\n(n=190)", "UIH\n(n=98)", "Test statistic", "P value"]
    rows = analysis["table_rows"]
    table = doc.add_table(rows=1, cols=6)
    for index, header in enumerate(headers):
        style_cell(table.rows[0].cells[index], header, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, size=8.0)
    set_repeat_table_header(table.rows[0])

    group_headers = {
        "Sex, n (%)",
        "Participant-level ONFH status, n (%)",
        "Maximum recorded ONFH stage among participants with ONFH, n (%)",
    }
    for item in rows:
        row = table.add_row()
        is_group = item["characteristic"] in group_headers
        values = [item[key] for key in ("characteristic", "overall", "siemens", "uih", "test", "p")]
        for index, value in enumerate(values):
            alignment = WD_ALIGN_PARAGRAPH.LEFT if index in (0, 4) else WD_ALIGN_PARAGRAPH.CENTER
            style_cell(row.cells[index], value, bold=is_group and index == 0, align=alignment, size=7.6, fill=PALE if is_group else None)
        if is_group:
            for cell in row.cells:
                set_cell_shading(cell, PALE)

    widths = [3650, 1700, 1750, 1750, 2500, 1610]
    set_table_geometry(table, widths)
    set_table_borders(table)

    note = (
        "Values are n (%) unless otherwise indicated. Age and sex were complete in both cohorts after the source record was updated; no missing-value imputation was performed. "
        "Age was compared using a two-sided Welch t-test; a two-sided Mann-Whitney U test was retained as a sensitivity analysis. Categorical variables were compared using two-sided Pearson chi-square tests; Fisher's exact test was additionally calculated for 2 x 2 tables. "
        "Exact P values are shown to three decimals, with P<0.001 reported as such. ONFH status and maximum stage were derived at participant level from bilateral hip labels to avoid treating paired hips as independent observations. "
        "Scanner model, magnetic field strength, receiver coil, and acquisition parameters were not retained in the supplied NRRD headers or JSON metadata and should be completed from original DICOM records before submission. "
        "ADR, source-filename token (expansion unavailable); FSE, fast spin echo; IQR, interquartile range; ONFH, osteonecrosis of the femoral head; SD, standard deviation; STIR, short tau inversion recovery; TIRM, turbo inversion recovery magnitude; TSE, turbo spin echo."
    )
    add_body(doc, note, size=7.4, color=MUTED, after=2)


def modal_text(mapping):
    key, count = max(mapping.items(), key=lambda item: item[1])
    total = sum(mapping.values())
    return f"{key} ({count}/{total}, {100 * count / total:.1f}%)"


def add_device_table(doc, analysis):
    add_heading(doc, "Device and supplied-image metadata", level=1)
    add_body(doc, "This table separates verified source metadata from fields that cannot be recovered from the supplied processed NRRD files.", size=9.2, color=MUTED, after=6)
    imaging = analysis["imaging_summary"]
    rows = [
        ("Manufacturer", "Siemens", "United Imaging Healthcare", "Verified from cohort directory/group definition"),
        ("Scanner model", "Not reported", "Not reported", "Requires original DICOM tag (0008,1090)"),
        ("Magnetic field strength", "Not reported", "Not reported", "Requires original DICOM tag (0018,0087)"),
        ("Receiver coil", "Not reported", "Not reported", "Requires original DICOM metadata"),
        ("T1 source filename label", "t1_tse_cor", "t1_fse_cor_ADR", "Filename-derived; ADR expansion unavailable"),
        ("T2 source filename label", "t2_tirm_cor", "t2_stir_cor", "Filename-derived"),
        ("Supplied left T1 matrix, mode", modal_text(imaging["SIEMENS"]["T1"]["matrix_sizes"]), modal_text(imaging["UIH"]["T1"]["matrix_sizes"]), "Processed NRRD header"),
        ("Supplied left T2 matrix, mode", modal_text(imaging["SIEMENS"]["T2"]["matrix_sizes"]), modal_text(imaging["UIH"]["T2"]["matrix_sizes"]), "Processed NRRD header"),
        ("Supplied left T1 voxel spacing, mode, mm", modal_text(imaging["SIEMENS"]["T1"]["voxel_spacings_mm"]), modal_text(imaging["UIH"]["T1"]["voxel_spacings_mm"]), "Processed NRRD geometry; not acquisition resolution"),
        ("Supplied left T2 voxel spacing, mode, mm", modal_text(imaging["SIEMENS"]["T2"]["voxel_spacings_mm"]), modal_text(imaging["UIH"]["T2"]["voxel_spacings_mm"]), "Processed NRRD geometry; not acquisition resolution"),
        ("Complete bilateral T1/T2 files", "190/190 (100%)", "98/98 (100%)", "All four JSON-referenced paths exist"),
    ]
    table = doc.add_table(rows=1, cols=4)
    headers = ["Field", "SIEMENS", "UIH", "Verification / limitation"]
    for index, header in enumerate(headers):
        style_cell(table.rows[0].cells[index], header, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, size=8.2)
    set_repeat_table_header(table.rows[0])
    for values in rows:
        row = table.add_row()
        for index, value in enumerate(values):
            style_cell(row.cells[index], value, align=WD_ALIGN_PARAGRAPH.LEFT, size=8.0)
    set_table_geometry(table, [3000, 3000, 3000, 3960])
    set_table_borders(table)
    add_body(doc, "No P value is appropriate for manufacturer, sequence naming, or scanner metadata because these characteristics define the device groups rather than represent exchangeable participant-level outcomes.", size=8.2, italic=True, color=MUTED, after=4)


def add_statistics_table(doc, analysis):
    add_heading(doc, "Statistical audit", level=1)
    tests = analysis["tests"]
    age = tests["age"]
    items = [
        ("Age", "Welch two-sample t-test", f"t({age['df']:.1f})={age['statistic']:.3f}", f"{age['p_value']:.3f}", f"Hedges g={age['hedges_g']:.3f}", f"Mann-Whitney U={age['sensitivity_u']:.1f}, P={age['sensitivity_p_value']:.3f}"),
        ("Sex", "Pearson chi-square", f"chi-square({tests['sex']['df']})={tests['sex']['statistic']:.3f}", f"{tests['sex']['p_value']:.3f}", f"Cramer's V={tests['sex']['cramers_v']:.3f}", f"Fisher exact P={tests['sex']['fisher_exact_p']:.3f}"),
        ("Participant ONFH status", "Pearson chi-square", f"chi-square({tests['onfh_status']['df']})={tests['onfh_status']['statistic']:.3f}", f"{tests['onfh_status']['p_value']:.3f}", f"Cramer's V={tests['onfh_status']['cramers_v']:.3f}", "Expected cell count minimum >5"),
        ("Maximum recorded stage", "Pearson chi-square", f"chi-square({tests['max_stage']['df']})={tests['max_stage']['statistic']:.3f}", f"{tests['max_stage']['p_value']:.3f}", f"Cramer's V={tests['max_stage']['cramers_v']:.3f}", "Among participants with ONFH"),
        ("Early/late involvement", "Pearson chi-square", f"chi-square({tests['early_late']['df']})={tests['early_late']['statistic']:.3f}", f"{tests['early_late']['p_value']:.3f}", f"Cramer's V={tests['early_late']['cramers_v']:.3f}", f"Fisher exact P={tests['early_late']['fisher_exact_p']:.3f}"),
    ]
    table = doc.add_table(rows=1, cols=6)
    headers = ["Variable", "Primary test", "Statistic", "P value", "Effect size", "Sensitivity / scope"]
    for index, header in enumerate(headers):
        style_cell(table.rows[0].cells[index], header, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, size=8.0)
    set_repeat_table_header(table.rows[0])
    for values in items:
        row = table.add_row()
        for index, value in enumerate(values):
            style_cell(row.cells[index], value, align=WD_ALIGN_PARAGRAPH.LEFT if index != 3 else WD_ALIGN_PARAGRAPH.CENTER, size=7.9)
    set_table_geometry(table, [2200, 2200, 2100, 1200, 1900, 3360])
    set_table_borders(table)

    age_desc = analysis["age_descriptive"]
    age_norm = analysis["age_normality"]
    notes = [
        f"Age completeness: SIEMENS {age_desc['SIEMENS']['n']}/190; UIH {age_desc['UIH']['n']}/98. No missing-value imputation was performed.",
        f"Age distribution checks: Shapiro-Wilk P={age_norm['SIEMENS']['p_value']:.3f} for SIEMENS and P={age_norm['UIH']['p_value']:.3f} for UIH.",
        f"Age mean difference (SIEMENS minus UIH): {age['mean_difference_years']:.2f} years (95% CI {age['mean_difference_95ci'][0]:.2f} to {age['mean_difference_95ci'][1]:.2f}).",
        "All tests are two-sided. No multiplicity adjustment was applied because the comparisons are descriptive/exploratory; interpretation should emphasize estimates and effect sizes rather than a binary significance threshold.",
        "Hip-level counts were not assigned naive P values because left and right hips from the same participant are correlated. Participant-level summaries were used for inference.",
    ]
    for note in notes:
        add_body(doc, note, size=8.6, after=3)


def add_provenance(doc):
    add_heading(doc, "Data provenance and reporting checklist", level=1)
    sources = [
        ("Demographics source", "E:\\3D_MedicalImageProcessing_for_Training\\ONFH_MRI_SIEMENS\\info_details.xlsx"),
        ("SIEMENS cohort definition", "E:\\3D_MedicalImageProcessing_for_Training\\ONFH_MRI_SIEMENS\\siemens_onfh_data.json"),
        ("UIH cohort definition", "E:\\3D_MedicalImageProcessing_for_Training\\ONFH_MRI_UIH\\uih_onfh_data_resampled.json"),
        ("Privacy check", "The two device cohorts contain no overlapping case IDs. Output tables exclude patient names and retain only aggregate values."),
        ("Before submission", "Retrieve scanner model, field strength, coil, repetition time, echo time, inversion time, flip angle, bandwidth, and original voxel size from the source DICOM headers or radiology protocol records."),
    ]
    for label, text in sources:
        paragraph = doc.add_paragraph()
        paragraph.paragraph_format.left_indent = Inches(0.18)
        paragraph.paragraph_format.first_line_indent = Inches(-0.18)
        paragraph.paragraph_format.space_before = Pt(0)
        paragraph.paragraph_format.space_after = Pt(4)
        paragraph.paragraph_format.line_spacing = 1.15
        set_run_font(paragraph.add_run(f"{label}: "), size=8.7, bold=True)
        set_run_font(paragraph.add_run(text), size=8.7)

    add_heading(doc, "Nature Portfolio alignment", level=2)
    add_body(doc, "The table reports the number of observations, central tendency and variability, exact P values, test statistics with degrees of freedom, effect sizes, missingness, and the analysis unit. This follows the reporting emphasis in Nature Portfolio statistical reporting materials and the npj guidance that table legends describe error analysis and ranges.", size=8.8, after=3)
    add_body(doc, "Journal guidance: https://www.nature.com/npjdigitalmed/for-authors-and-referees/guide-to-authors", size=8.2, color=MUTED, after=2)
    add_body(doc, "npj table guidance: https://www.nature.com/documents/npjdigitalmed-gta.pdf", size=8.2, color=MUTED, after=2)
    add_body(doc, "Nature Portfolio reporting summary example: https://www.nature.com/articles/s41586-023-05772-8.pdf", size=8.2, color=MUTED, after=2)


def configure_document(doc):
    section = doc.sections[0]
    section.orientation = WD_ORIENT.LANDSCAPE
    section.page_width = Inches(11)
    section.page_height = Inches(8.5)
    section.top_margin = Inches(0.72)
    section.bottom_margin = Inches(0.72)
    section.left_margin = Inches(1)
    section.right_margin = Inches(1)
    section.header_distance = Inches(0.35)
    section.footer_distance = Inches(0.35)

    styles = doc.styles
    normal = styles["Normal"]
    normal.font.name = "Calibri"
    normal._element.rPr.rFonts.set(qn("w:ascii"), "Calibri")
    normal._element.rPr.rFonts.set(qn("w:hAnsi"), "Calibri")
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
    normal.font.size = Pt(11)
    normal.paragraph_format.space_before = Pt(0)
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.25

    for level, size, before, after, color in ((1, 16, 18, 10, "2E74B5"), (2, 13, 14, 7, "2E74B5"), (3, 12, 10, 5, "1F4D78")):
        style = styles[f"Heading {level}"]
        style.font.name = "Calibri"
        style._element.rPr.rFonts.set(qn("w:ascii"), "Calibri")
        style._element.rPr.rFonts.set(qn("w:hAnsi"), "Calibri")
        style._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = RGBColor.from_string(color)
        style.paragraph_format.space_before = Pt(before)
        style.paragraph_format.space_after = Pt(after)
        style.paragraph_format.keep_with_next = True

    list_style = styles["List Bullet"]
    list_style.font.name = "Calibri"
    list_style.font.size = Pt(8.7)
    list_style.paragraph_format.left_indent = Inches(0.375)
    list_style.paragraph_format.first_line_indent = Inches(-0.188)
    list_style.paragraph_format.space_after = Pt(4)
    list_style.paragraph_format.line_spacing = 1.25

    header = section.header.paragraphs[0]
    header.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    header.paragraph_format.space_after = Pt(0)
    set_run_font(header.add_run("ONFH cohort | demographic table"), size=7.5, color=MUTED, name="Arial")
    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    footer.paragraph_format.space_after = Pt(0)
    set_run_font(footer.add_run("Prepared for manuscript development"), size=7.2, color=MUTED, name="Arial")


def main():
    analysis = json.loads(ANALYSIS_JSON.read_text(encoding="utf-8"))
    document = Document()
    configure_document(document)
    add_title_block(document)
    add_main_table(document, analysis)
    document.add_page_break()
    add_device_table(document, analysis)
    document.add_page_break()
    add_statistics_table(document, analysis)
    document.add_page_break()
    add_provenance(document)
    document.core_properties.title = "Demographic and imaging characteristics by MRI device group"
    document.core_properties.subject = "Nature Portfolio-style ONFH cohort table"
    document.core_properties.author = ""
    document.core_properties.keywords = "ONFH; MRI; demographics; SIEMENS; UIH"
    document.save(OUTPUT_DOCX)
    print(OUTPUT_DOCX.resolve())


if __name__ == "__main__":
    main()
