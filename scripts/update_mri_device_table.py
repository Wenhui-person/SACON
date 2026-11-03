from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt, RGBColor


DOCX = Path("outputs/demographics_npj_nature/Table1_demographics_npj_nature.docx")


def set_cell_text(cell, text: str, *, bold: bool = False, align=WD_ALIGN_PARAGRAPH.LEFT):
    cell.text = ""
    p = cell.paragraphs[0]
    p.alignment = align
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(0)
    p.paragraph_format.line_spacing = 1.0
    r = p.add_run(text)
    r.font.name = "Arial"
    r._element.get_or_add_rPr().rFonts.set(qn("w:ascii"), "Arial")
    r._element.get_or_add_rPr().rFonts.set(qn("w:hAnsi"), "Arial")
    r._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), "Arial")
    r.font.size = Pt(7.5)
    r.bold = bold
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER


def set_shading(cell, fill: str):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)
    shd.set(qn("w:val"), "clear")


def ensure_repeat_header(row):
    tr_pr = row._tr.get_or_add_trPr()
    tag = tr_pr.find(qn("w:tblHeader"))
    if tag is None:
        tag = OxmlElement("w:tblHeader")
        tr_pr.append(tag)
    tag.set(qn("w:val"), "true")


def replace_table_body(table, rows):
    template_tr = deepcopy(table.rows[1]._tr)
    for row in list(table.rows)[1:]:
        table._tbl.remove(row._tr)
    for idx, values in enumerate(rows, 1):
        tr = deepcopy(template_tr)
        table._tbl.append(tr)
        row = table.rows[-1]
        for ci, (cell, value) in enumerate(zip(row.cells, values)):
            set_cell_text(cell, value, bold=(ci == 0))
            set_shading(cell, "F5F8FA" if idx % 2 == 0 else "FFFFFF")
    ensure_repeat_header(table.rows[0])


def replace_sentence(paragraph, old: str, new: str):
    text = paragraph.text
    if new in text:
        return
    if old not in text:
        raise ValueError(f"Expected sentence not found: {old}")
    paragraph.text = text.replace(old, new)
    for run in paragraph.runs:
        run.font.name = "Arial"
        run._element.get_or_add_rPr().rFonts.set(qn("w:ascii"), "Arial")
        run._element.get_or_add_rPr().rFonts.set(qn("w:hAnsi"), "Arial")
        run.font.size = Pt(8)


def main():
    doc = Document(DOCX)

    summary_cell = doc.tables[0].rows[0].cells[3]
    summary_cell.text = ""
    summary_p = summary_cell.paragraphs[0]
    label = summary_p.add_run("GENERATED")
    label.font.name = "Arial"
    label.font.size = Pt(8)
    label.bold = True
    label.font.color.rgb = RGBColor(91, 102, 119)
    label.add_break()
    value = summary_p.add_run("10 Sep 2026")
    value.font.name = "Arial"
    value.font.size = Pt(10)
    value.bold = True

    main_table = doc.tables[1]
    main_updates = {
        19: ("MRI manufacturer", "Two vendors", "Siemens", "United Imaging Healthcare", "Not applicable", "—"),
        20: ("T1-weighted acquisition", "2D coronal spin echo", "Turbo spin echo (TSE)", "Fast spin echo (FSE; ADR)", "Descriptive", "—"),
        22: ("Scanner model / field strength", "Two platforms", "Verio / 3 T", "uMR 660 / 1.5 T", "Not applicable", "—"),
    }
    for ri, values in main_updates.items():
        for ci, value in enumerate(values):
            set_cell_text(main_table.rows[ri].cells[ci], value, bold=(ci == 0))

    old_claim = (
        "Scanner model, magnetic field strength, receiver coil, and acquisition parameters were not retained "
        "in the supplied NRRD headers or JSON metadata and should be completed from original DICOM records "
        "before submission."
    )
    new_claim = (
        "Scanner model, field strength, software version, and representative T1-weighted acquisition "
        "parameters were obtained from original DICOM headers. A deterministic random audit of 16 examination "
        "folders was used to confirm tag interpretation and assess protocol variability; it was not used to "
        "estimate cohort-wide parameter frequencies."
    )
    replace_sentence(doc.paragraphs[5], old_claim, new_claim)

    doc.paragraphs[8].text = (
        "The two study groups were acquired on different vendor platforms and field strengths (Siemens Verio, "
        "3 T; United Imaging Healthcare uMR 660, 1.5 T). Differences in echo time, slice geometry, and in-plane "
        "sampling constitute potential device-domain effects and should be considered in model validation."
    )
    doc.paragraphs[9].text = (
        "No P value is appropriate for scanner or protocol descriptors because they define the acquisition "
        "domains rather than exchangeable participant-level outcomes."
    )

    device_rows = [
        ("Manufacturer", "Siemens", "United Imaging Healthcare (UIH)", "Direct DICOM equipment identity"),
        ("Scanner model", "Verio", "uMR 660", "DICOM (0008,1090); confirmed in sampled series"),
        ("Magnetic field strength, T", "3.0", "1.5", "DICOM (0018,0087); fixed by platform"),
        ("Software version", "syngo MR B19", "R001", "Direct DICOM software identifier"),
        ("Acquisition site", "Xi'an Honghui Hospital", "Xi'an Honghui Hospital, North Campus", "Campus-level reporting; street addresses omitted"),
        ("T1 series / protocol label", "t1_tse_cor_p2_512", "t1_fse_cor_ADR", "Vendor protocol labels retained verbatim"),
        ("Acquisition plane / type", "Coronal / 2D", "Coronal / 2D", "Plane inferred from protocol label; type from DICOM"),
        ("Pulse sequence", "Spin echo; turbo spin echo", "Spin echo; fast spin echo", "TSE and FSE are vendor terms for related refocused echo-train methods"),
        ("Sequence implementation", "*tse2d1_4; SK/SP/OSP", "fse; variant NONE", "Vendor-specific identifiers; not harmonized"),
        ("Repetition time (TR), ms", "881", "398", "Representative supplied protocol values"),
        ("Echo time (TE), ms", "20.00", "8.28", "Representative supplied protocol values"),
        ("Inversion time (TI)", "Not applicable", "Not applicable", "No inversion-recovery preparation reported"),
        ("Flip angle, degrees", "150", "150", "DICOM Flip Angle (0018,1314)"),
        ("Slice thickness, mm", "3.5", "4.0", "Nominal reconstructed slice thickness"),
        ("Slice spacing, mm", "5.425", "4.400", "Center-to-center spacing; not slice thickness"),
        ("Calculated interslice gap, mm", "1.925", "0.400", "Slice spacing minus slice thickness"),
        ("In-plane pixel spacing, mm", "0.893 × 0.893", "0.521 × 0.521", "DICOM Pixel Spacing; representative protocol values"),
        ("Patient position", "Head first-supine (HFS)", "Head first-supine (HFS)", "Consistent in the sampled target-platform series"),
        ("Body part", "Not recorded", "Hip", "Blank Siemens tag should not be interpreted as missing anatomy"),
        ("Transmit / receive coil", "Body / not recorded", "VTC / not recorded", "Receiver-coil tag was blank; VTC retained as recorded"),
        ("Random DICOM audit", "Verio: 4 coronal T1 series", "uMR 660: 6 coronal T1 series", "16 folders sampled (seed 20260910); parameter variation was observed"),
        ("Audit interpretation", "TR 366–881 ms; TE 9–20 ms; spacing 4.550–5.425 mm", "TR 379–398 ms; TE 8.28–8.68 ms; thickness 3.0–4.5 mm", "Ranges describe sampled series only, not the full analytic cohorts"),
    ]
    replace_table_body(doc.tables[2], device_rows)

    doc.paragraphs[10].text = (
        "Notes: ADR is retained as a vendor protocol suffix because its expansion is not encoded in the standard "
        "DICOM fields reviewed. Device serial numbers, workstation names, and full postal addresses are omitted "
        "because they do not improve scientific interpretation and may increase site identifiability. Blank "
        "receiver-coil fields are reported as not recorded rather than inferred."
    )
    doc.paragraphs[23].text = (
        "DICOM verification: Scanner identity and representative T1 protocol parameters were checked directly "
        "against original DICOM headers in a deterministic random sample of 16 examination folders. Because "
        "protocol variation was present, the listed TR, TE, slice spacing, and pixel spacing should be described "
        "as representative values unless a full-cohort DICOM audit is completed."
    )

    doc.paragraphs[7].text = "MRI acquisition equipment and protocol metadata"
    doc.paragraphs[9].paragraph_format.keep_with_next = True
    doc.paragraphs[10].paragraph_format.keep_with_next = True

    for pi in (8, 9, 10, 23):
        p = doc.paragraphs[pi]
        p.paragraph_format.space_after = Pt(5)
        p.paragraph_format.line_spacing = 1.05
        for run in p.runs:
            run.font.name = "Arial"
            run._element.get_or_add_rPr().rFonts.set(qn("w:ascii"), "Arial")
            run._element.get_or_add_rPr().rFonts.set(qn("w:hAnsi"), "Arial")
            run.font.size = Pt(8)

    tmp = DOCX.with_name(DOCX.stem + ".updated.docx")
    doc.save(tmp)
    tmp.replace(DOCX)
    print(DOCX.resolve())


if __name__ == "__main__":
    main()
