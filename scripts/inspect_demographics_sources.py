from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import openpyxl


ROOTS = {
    "SIEMENS": Path(r"E:\3D_MedicalImageProcessing_for_Training\ONFH_MRI_SIEMENS"),
    "UIH": Path(r"E:\3D_MedicalImageProcessing_for_Training\ONFH_MRI_UIH"),
}


def read_nrrd_header(path: Path) -> list[str]:
    with path.open("rb") as stream:
        payload = stream.read(65536)
    header = payload.split(b"\n\n", 1)[0]
    return header.decode("utf-8", errors="replace").splitlines()


def workbook_rows(path: Path) -> list[dict]:
    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    sheet = workbook.active
    rows = sheet.iter_rows(values_only=True)
    headers = [str(value).strip() if value is not None else "" for value in next(rows)]
    return [dict(zip(headers, row)) for row in rows]


def main() -> None:
    all_rows = workbook_rows(ROOTS["SIEMENS"] / "info_details.xlsx")
    row_by_id = {str(row["case_id"]).strip(): row for row in all_rows if row.get("case_id")}
    print("MASTER_ROWS", len(all_rows), "UNIQUE_IDS", len(row_by_id))

    group_ids = {}
    for group, root in ROOTS.items():
        json_path = next(root.glob("*.json"))
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        ids = [str(value).strip() for value in payload["patient_id"]]
        group_ids[group] = ids
        print("\nGROUP", group, "N", len(ids), "UNIQUE", len(set(ids)))
        print("FIRST_LAST", ids[:3], ids[-3:])
        print("MISSING_FROM_MASTER", sorted(set(ids) - set(row_by_id))[:20], len(set(ids) - set(row_by_id)))
        sex = Counter(row_by_id.get(case_id, {}).get("Sex") for case_id in ids)
        ages = [row_by_id.get(case_id, {}).get("Age") for case_id in ids]
        print("SEX_RAW_REPR", {repr(k): v for k, v in sex.items()})
        print("AGE_MISSING", sum(value is None for value in ages), "AGE_VALUES", ages[:10])
        print("PATH_KEYS", {key: len(value) for key, value in payload.items()})

        sample = next(root.rglob("*.nrrd"))
        print("NRRD_SAMPLE", sample)
        for line in read_nrrd_header(sample):
            low = line.lower()
            if any(token in low for token in ("manufacturer", "model", "field", "magnetic", "dicom", "modality", "sequence", "space", "sizes", "spacings")):
                print("HDR", line)

    overlap = set(group_ids["SIEMENS"]) & set(group_ids["UIH"])
    print("\nOVERLAP", len(overlap), sorted(overlap)[:30])


if __name__ == "__main__":
    main()
