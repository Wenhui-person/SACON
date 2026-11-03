from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import openpyxl
from scipy import stats


MASTER = Path(r"E:\3D_MedicalImageProcessing_for_Training\ONFH_MRI_SIEMENS\info_details.xlsx")
JSONS = {
    "SIEMENS": Path(r"E:\3D_MedicalImageProcessing_for_Training\ONFH_MRI_SIEMENS\siemens_onfh_data.json"),
    "UIH": Path(r"E:\3D_MedicalImageProcessing_for_Training\ONFH_MRI_UIH\uih_onfh_data_resampled.json"),
}
ANALYSIS = Path("outputs/demographics_npj_nature/demographic_analysis.json")


def close(a: float, b: float, tolerance: float = 1e-12) -> None:
    if not math.isclose(a, b, rel_tol=tolerance, abs_tol=tolerance):
        raise AssertionError(f"Mismatch: {a!r} != {b!r}")


def main() -> None:
    worksheet = openpyxl.load_workbook(MASTER, read_only=True, data_only=True).active
    source = worksheet.iter_rows(values_only=True)
    headers = next(source)
    rows = {row[0]: dict(zip(headers, row)) for row in source if row[0]}
    records: dict[str, list[dict]] = {}

    for group, path in JSONS.items():
        payload = json.loads(path.read_text(encoding="utf-8"))
        group_rows = []
        for index, case_id in enumerate(payload["patient_id"]):
            source_row = rows[case_id]
            group_rows.append(
                {
                    "case_id": case_id,
                    "age": float(source_row["Age"]),
                    "sex": source_row["Sex"],
                    "left_label": int(payload["label_left_bin"][index]),
                    "right_label": int(payload["label_right_bin"][index]),
                    "left_stage": int(payload["label_left_stage"][index]),
                    "right_stage": int(payload["label_right_stage"][index]),
                    "left_els": int(payload["label_left_ELS"][index]),
                    "right_els": int(payload["label_right_ELS"][index]),
                }
            )
        records[group] = group_rows

    assert len(records["SIEMENS"]) == 190
    assert len(records["UIH"]) == 98
    assert not ({row["case_id"] for row in records["SIEMENS"]} & {row["case_id"] for row in records["UIH"]})
    assert all(row["age"] is not None and row["sex"] in {"男", "女"} for values in records.values() for row in values)
    completed = next(row for row in records["UIH"] if row["case_id"] == "mri20266240001")
    assert completed["age"] == 45 and completed["sex"] == "男"

    ages_1 = np.array([row["age"] for row in records["SIEMENS"]], dtype=float)
    ages_2 = np.array([row["age"] for row in records["UIH"]], dtype=float)
    variance_1 = ages_1.var(ddof=1)
    variance_2 = ages_2.var(ddof=1)
    standard_error = math.sqrt(variance_1 / len(ages_1) + variance_2 / len(ages_2))
    t_manual = (ages_1.mean() - ages_2.mean()) / standard_error
    df_manual = (variance_1 / len(ages_1) + variance_2 / len(ages_2)) ** 2 / (
        (variance_1 / len(ages_1)) ** 2 / (len(ages_1) - 1)
        + (variance_2 / len(ages_2)) ** 2 / (len(ages_2) - 1)
    )
    p_manual = 2 * stats.t.sf(abs(t_manual), df_manual)
    t_scipy = stats.ttest_ind(ages_1, ages_2, equal_var=False)
    close(t_manual, float(t_scipy.statistic))
    close(p_manual, float(t_scipy.pvalue))

    sex_table = np.array(
        [
            [sum(row["sex"] == "男" for row in records["SIEMENS"]), sum(row["sex"] == "女" for row in records["SIEMENS"])],
            [sum(row["sex"] == "男" for row in records["UIH"]), sum(row["sex"] == "女" for row in records["UIH"])],
        ],
        dtype=int,
    )
    expected = np.outer(sex_table.sum(axis=1), sex_table.sum(axis=0)) / sex_table.sum()
    chi2_manual = float(np.sum((sex_table - expected) ** 2 / expected))
    p_chi_manual = float(stats.chi2.sf(chi2_manual, 1))
    chi2_scipy, p_chi_scipy, _, expected_scipy = stats.chi2_contingency(sex_table, correction=False)
    close(chi2_manual, float(chi2_scipy))
    close(p_chi_manual, float(p_chi_scipy))
    if not np.allclose(expected, expected_scipy):
        raise AssertionError("Expected-frequency calculation mismatch")

    saved = json.loads(ANALYSIS.read_text(encoding="utf-8"))
    close(t_manual, saved["tests"]["age"]["statistic"])
    close(df_manual, saved["tests"]["age"]["df"])
    close(p_manual, saved["tests"]["age"]["p_value"])
    close(chi2_manual, saved["tests"]["sex"]["statistic"])
    close(p_chi_manual, saved["tests"]["sex"]["p_value"])
    assert saved["missing"] == {"SIEMENS": {"age": 0, "sex": 0}, "UIH": {"age": 0, "sex": 0}}

    print(
        json.dumps(
            {
                "completed_case": {"case_id": completed["case_id"], "age": completed["age"], "sex": completed["sex"]},
                "age": {
                    "siemens_n": len(ages_1),
                    "uih_n": len(ages_2),
                    "siemens_mean": ages_1.mean(),
                    "uih_mean": ages_2.mean(),
                    "standard_error": standard_error,
                    "t": t_manual,
                    "df": df_manual,
                    "p": p_manual,
                },
                "sex": {"observed": sex_table.tolist(), "expected": expected.tolist(), "chi2": chi2_manual, "p": p_chi_manual},
                "verification": "PASS",
            },
            ensure_ascii=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
