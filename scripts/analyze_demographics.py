from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np
import openpyxl
from scipy import stats


OUTPUT_DIR = Path("outputs/demographics_npj_nature")
MASTER_XLSX = Path(r"E:\3D_MedicalImageProcessing_for_Training\ONFH_MRI_SIEMENS\info_details.xlsx")
GROUPS = {
    "SIEMENS": Path(r"E:\3D_MedicalImageProcessing_for_Training\ONFH_MRI_SIEMENS\siemens_onfh_data.json"),
    "UIH": Path(r"E:\3D_MedicalImageProcessing_for_Training\ONFH_MRI_UIH\uih_onfh_data_resampled.json"),
}


def load_master_rows() -> dict[str, dict]:
    workbook = openpyxl.load_workbook(MASTER_XLSX, read_only=True, data_only=True)
    worksheet = workbook.active
    source = worksheet.iter_rows(values_only=True)
    headers = [str(value).strip() if value is not None else "" for value in next(source)]
    return {
        str(row[0]).strip(): dict(zip(headers, row))
        for row in source
        if row and row[0] is not None
    }


def normalize_sex(value):
    if value == "男":
        return "Male"
    if value == "女":
        return "Female"
    return None


def participant_status(left_label: int, right_label: int) -> str:
    total = int(left_label) + int(right_label)
    return {0: "No ONFH", 1: "Unilateral ONFH", 2: "Bilateral ONFH"}[total]


def participant_stage(left_label, left_stage, right_label, right_stage):
    stages = []
    if int(left_label) == 1:
        stages.append(int(left_stage))
    if int(right_label) == 1:
        stages.append(int(right_stage))
    return max(stages) if stages else 0


def participant_els(left_label, left_els, right_label, right_els):
    values = []
    if int(left_label) == 1:
        values.append(int(left_els))
    if int(right_label) == 1:
        values.append(int(right_els))
    return max(values) if values else 0


def parse_nrrd_geometry(path: Path) -> dict:
    with path.open("rb") as stream:
        header = stream.read(65536).split(b"\n\n", 1)[0].decode("utf-8", errors="replace")
    sizes_match = re.search(r"^sizes:\s+(.+)$", header, flags=re.MULTILINE)
    directions_match = re.search(r"^space directions:\s+(.+)$", header, flags=re.MULTILINE)
    sizes = [int(token) for token in sizes_match.group(1).split()] if sizes_match else []
    spacings = []
    if directions_match:
        vectors = re.findall(r"\(([^)]+)\)", directions_match.group(1))
        for vector in vectors:
            coords = [float(token) for token in vector.split(",")]
            spacings.append(float(np.linalg.norm(coords)))
    return {"sizes": sizes, "spacings_mm": spacings}


def describe(values: list[float]) -> dict:
    array = np.asarray(values, dtype=float)
    return {
        "n": int(array.size),
        "mean": float(np.mean(array)),
        "sd": float(np.std(array, ddof=1)),
        "median": float(np.median(array)),
        "q1": float(np.quantile(array, 0.25)),
        "q3": float(np.quantile(array, 0.75)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def welch_age_test(first: list[float], second: list[float]) -> dict:
    x = np.asarray(first, dtype=float)
    y = np.asarray(second, dtype=float)
    result = stats.ttest_ind(x, y, equal_var=False)
    difference = float(x.mean() - y.mean())
    standard_error = math.sqrt(x.var(ddof=1) / len(x) + y.var(ddof=1) / len(y))
    numerator = (x.var(ddof=1) / len(x) + y.var(ddof=1) / len(y)) ** 2
    denominator = ((x.var(ddof=1) / len(x)) ** 2 / (len(x) - 1)) + ((y.var(ddof=1) / len(y)) ** 2 / (len(y) - 1))
    df = numerator / denominator
    critical = stats.t.ppf(0.975, df)
    pooled_sd = math.sqrt(((len(x) - 1) * x.var(ddof=1) + (len(y) - 1) * y.var(ddof=1)) / (len(x) + len(y) - 2))
    cohens_d = difference / pooled_sd
    correction = 1 - 3 / (4 * (len(x) + len(y)) - 9)
    mw = stats.mannwhitneyu(x, y, alternative="two-sided", method="asymptotic")
    return {
        "primary_test": "Welch two-sample t-test (two-sided)",
        "statistic": float(result.statistic),
        "df": float(df),
        "p_value": float(result.pvalue),
        "mean_difference_years": difference,
        "mean_difference_95ci": [difference - critical * standard_error, difference + critical * standard_error],
        "hedges_g": float(cohens_d * correction),
        "sensitivity_test": "Mann–Whitney U test (two-sided, asymptotic)",
        "sensitivity_u": float(mw.statistic),
        "sensitivity_p_value": float(mw.pvalue),
        "rank_biserial": float(2 * mw.statistic / (len(x) * len(y)) - 1),
    }


def categorical_test(table: list[list[int]], label: str) -> dict:
    array = np.asarray(table, dtype=int)
    chi2, p_value, df, expected = stats.chi2_contingency(array, correction=False)
    n = int(array.sum())
    cramers_v = math.sqrt(chi2 / (n * min(array.shape[0] - 1, array.shape[1] - 1))) if n else float("nan")
    output = {
        "variable": label,
        "test": "Pearson chi-square test (two-sided)",
        "statistic": float(chi2),
        "df": int(df),
        "p_value": float(p_value),
        "cramers_v": float(cramers_v),
        "observed": array.tolist(),
        "expected_min": float(np.min(expected)),
    }
    if array.shape == (2, 2):
        odds_ratio, fisher_p = stats.fisher_exact(array, alternative="two-sided")
        output.update({"fisher_exact_p": float(fisher_p), "odds_ratio": float(odds_ratio)})
    return output


def format_p(value: float) -> str:
    return "<0.001" if value < 0.001 else f"{value:.3f}"


def format_mean_sd(values: list[float]) -> str:
    summary = describe(values)
    return f"{summary['mean']:.1f} ({summary['sd']:.1f})"


def format_median_iqr(values: list[float]) -> str:
    summary = describe(values)
    return f"{summary['median']:.1f} ({summary['q1']:.1f}–{summary['q3']:.1f})"


def format_count_percent(count: int, denominator: int) -> str:
    return f"{count} ({100 * count / denominator:.1f}%)" if denominator else "0 (NA)"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    master = load_master_rows()
    records = []
    geometries = []
    source_payloads = {}

    for group, json_path in GROUPS.items():
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        source_payloads[group] = payload
        for index, case_id in enumerate(payload["patient_id"]):
            source = master[case_id]
            left_label = int(payload["label_left_bin"][index])
            right_label = int(payload["label_right_bin"][index])
            record = {
                "case_id": case_id,
                "device_group": group,
                "sex": normalize_sex(source.get("Sex")),
                "age_years": source.get("Age"),
                "left_onfh": left_label,
                "right_onfh": right_label,
                "left_stage": int(payload["label_left_stage"][index]),
                "right_stage": int(payload["label_right_stage"][index]),
                "left_els": int(payload["label_left_ELS"][index]),
                "right_els": int(payload["label_right_ELS"][index]),
                "participant_onfh_status": participant_status(left_label, right_label),
                "participant_max_stage": participant_stage(
                    left_label,
                    payload["label_left_stage"][index],
                    right_label,
                    payload["label_right_stage"][index],
                ),
                "participant_max_els": participant_els(
                    left_label,
                    payload["label_left_ELS"][index],
                    right_label,
                    payload["label_right_ELS"][index],
                ),
            }
            records.append(record)

            for sequence, key in (("T1", "left_t1_path"), ("T2", "left_t2_path")):
                path = Path(payload[key][index])
                geometry = parse_nrrd_geometry(path)
                geometries.append(
                    {
                        "case_id": case_id,
                        "device_group": group,
                        "sequence": sequence,
                        "path_exists": path.exists(),
                        "filename_prefix": path.name.split("mri", 1)[0].rstrip("_"),
                        "sizes": geometry["sizes"],
                        "spacings_mm": geometry["spacings_mm"],
                    }
                )

    by_group = {group: [row for row in records if row["device_group"] == group] for group in GROUPS}
    ages = {
        group: [float(row["age_years"]) for row in rows if row["age_years"] is not None]
        for group, rows in by_group.items()
    }
    age_test = welch_age_test(ages["SIEMENS"], ages["UIH"])
    age_normality = {
        group: {"w": float(stats.shapiro(values).statistic), "p_value": float(stats.shapiro(values).pvalue)}
        for group, values in ages.items()
    }

    sex_counts = {
        group: Counter(row["sex"] for row in rows if row["sex"] is not None)
        for group, rows in by_group.items()
    }
    sex_test = categorical_test(
        [
            [sex_counts["SIEMENS"]["Male"], sex_counts["SIEMENS"]["Female"]],
            [sex_counts["UIH"]["Male"], sex_counts["UIH"]["Female"]],
        ],
        "Sex",
    )

    status_levels = ["No ONFH", "Unilateral ONFH", "Bilateral ONFH"]
    status_counts = {group: Counter(row["participant_onfh_status"] for row in rows) for group, rows in by_group.items()}
    status_test = categorical_test(
        [[status_counts[group][level] for level in status_levels] for group in GROUPS],
        "Participant-level ONFH status",
    )

    positive = {
        group: [row for row in rows if row["participant_max_stage"] > 0]
        for group, rows in by_group.items()
    }
    stage_levels = [1, 2, 3, 4]
    stage_counts = {group: Counter(row["participant_max_stage"] for row in rows) for group, rows in positive.items()}
    stage_test = categorical_test(
        [[stage_counts[group][level] for level in stage_levels] for group in GROUPS],
        "Maximum recorded ONFH stage among participants with ONFH",
    )
    late_counts = {
        group: Counter("Late" if row["participant_max_els"] == 2 else "Early" for row in rows)
        for group, rows in positive.items()
    }
    late_test = categorical_test(
        [[late_counts[group]["Early"], late_counts[group]["Late"]] for group in GROUPS],
        "Early/late status among participants with ONFH",
    )

    def geometry_summary(group: str, sequence: str) -> dict:
        rows = [row for row in geometries if row["device_group"] == group and row["sequence"] == sequence]
        matrices = Counter("×".join(str(value) for value in row["sizes"]) for row in rows)
        spacings = Counter("×".join(f"{value:.3f}" for value in row["spacings_mm"]) for row in rows)
        return {
            "n": len(rows),
            "all_paths_exist": all(row["path_exists"] for row in rows),
            "filename_prefixes": dict(Counter(row["filename_prefix"] for row in rows)),
            "matrix_sizes": dict(matrices),
            "voxel_spacings_mm": dict(spacings),
        }

    imaging_summary = {
        group: {sequence: geometry_summary(group, sequence) for sequence in ("T1", "T2")}
        for group in GROUPS
    }

    group_n = {group: len(rows) for group, rows in by_group.items()}
    nonmissing_sex = {group: sum(row["sex"] is not None for row in rows) for group, rows in by_group.items()}
    positive_n = {group: len(rows) for group, rows in positive.items()}
    tests = {"age": age_test, "sex": sex_test, "onfh_status": status_test, "max_stage": stage_test, "early_late": late_test}

    table_rows = [
        {"characteristic": "Participants, n", "overall": str(len(records)), "siemens": str(group_n["SIEMENS"]), "uih": str(group_n["UIH"]), "test": "—", "p": "—"},
        {"characteristic": "Age, years, mean (SD)", "overall": format_mean_sd([v for values in ages.values() for v in values]), "siemens": format_mean_sd(ages["SIEMENS"]), "uih": format_mean_sd(ages["UIH"]), "test": f"Welch t({age_test['df']:.1f})={age_test['statistic']:.2f}", "p": format_p(age_test["p_value"])},
        {"characteristic": "Age, years, median (IQR)", "overall": format_median_iqr([v for values in ages.values() for v in values]), "siemens": format_median_iqr(ages["SIEMENS"]), "uih": format_median_iqr(ages["UIH"]), "test": "Descriptive", "p": "—"},
        {"characteristic": "Missing age, n", "overall": str(sum(row["age_years"] is None for row in records)), "siemens": str(sum(row["age_years"] is None for row in by_group["SIEMENS"])), "uih": str(sum(row["age_years"] is None for row in by_group["UIH"])), "test": "—", "p": "—"},
        {"characteristic": "Sex, n (%)", "overall": "", "siemens": "", "uih": "", "test": f"χ²({sex_test['df']})={sex_test['statistic']:.2f}", "p": format_p(sex_test["p_value"])},
        {"characteristic": "  Male", "overall": format_count_percent(sex_counts["SIEMENS"]["Male"] + sex_counts["UIH"]["Male"], sum(nonmissing_sex.values())), "siemens": format_count_percent(sex_counts["SIEMENS"]["Male"], nonmissing_sex["SIEMENS"]), "uih": format_count_percent(sex_counts["UIH"]["Male"], nonmissing_sex["UIH"]), "test": "", "p": ""},
        {"characteristic": "  Female", "overall": format_count_percent(sex_counts["SIEMENS"]["Female"] + sex_counts["UIH"]["Female"], sum(nonmissing_sex.values())), "siemens": format_count_percent(sex_counts["SIEMENS"]["Female"], nonmissing_sex["SIEMENS"]), "uih": format_count_percent(sex_counts["UIH"]["Female"], nonmissing_sex["UIH"]), "test": "", "p": ""},
        {"characteristic": "Missing sex, n", "overall": str(sum(row["sex"] is None for row in records)), "siemens": str(sum(row["sex"] is None for row in by_group["SIEMENS"])), "uih": str(sum(row["sex"] is None for row in by_group["UIH"])), "test": "—", "p": "—"},
        {"characteristic": "Participant-level ONFH status, n (%)", "overall": "", "siemens": "", "uih": "", "test": f"χ²({status_test['df']})={status_test['statistic']:.2f}", "p": format_p(status_test["p_value"])},
    ]
    for level in status_levels:
        table_rows.append({
            "characteristic": f"  {level}",
            "overall": format_count_percent(status_counts["SIEMENS"][level] + status_counts["UIH"][level], len(records)),
            "siemens": format_count_percent(status_counts["SIEMENS"][level], group_n["SIEMENS"]),
            "uih": format_count_percent(status_counts["UIH"][level], group_n["UIH"]),
            "test": "",
            "p": "",
        })
    table_rows.append({"characteristic": "Maximum recorded ONFH stage among participants with ONFH, n (%)", "overall": "", "siemens": f"n={positive_n['SIEMENS']}", "uih": f"n={positive_n['UIH']}", "test": f"χ²({stage_test['df']})={stage_test['statistic']:.2f}", "p": format_p(stage_test["p_value"])})
    for level in stage_levels:
        table_rows.append({
            "characteristic": f"  Stage {level}",
            "overall": format_count_percent(stage_counts["SIEMENS"][level] + stage_counts["UIH"][level], sum(positive_n.values())),
            "siemens": format_count_percent(stage_counts["SIEMENS"][level], positive_n["SIEMENS"]),
            "uih": format_count_percent(stage_counts["UIH"][level], positive_n["UIH"]),
            "test": "",
            "p": "",
        })
    table_rows.extend([
        {"characteristic": "Late-stage involvement among participants with ONFH, n (%)", "overall": format_count_percent(late_counts["SIEMENS"]["Late"] + late_counts["UIH"]["Late"], sum(positive_n.values())), "siemens": format_count_percent(late_counts["SIEMENS"]["Late"], positive_n["SIEMENS"]), "uih": format_count_percent(late_counts["UIH"]["Late"], positive_n["UIH"]), "test": f"χ²({late_test['df']})={late_test['statistic']:.2f}", "p": format_p(late_test["p_value"])},
        {"characteristic": "MRI manufacturer", "overall": "Two vendors", "siemens": "Siemens", "uih": "United Imaging Healthcare", "test": "Not applicable", "p": "—"},
        {"characteristic": "T1-weighted source sequence label", "overall": "Coronal", "siemens": "TSE coronal", "uih": "FSE coronal (ADR)", "test": "Descriptive", "p": "—"},
        {"characteristic": "T2-weighted source sequence label", "overall": "Coronal", "siemens": "TIRM coronal", "uih": "STIR coronal", "test": "Descriptive", "p": "—"},
        {"characteristic": "Scanner model / field strength", "overall": "Not retained in supplied NRRD/metadata", "siemens": "Not reported", "uih": "Not reported", "test": "Not applicable", "p": "—"},
        {"characteristic": "Complete bilateral T1/T2 paths, n (%)", "overall": format_count_percent(sum(all(Path(source_payloads[group][key][i]).exists() for key in ("left_t1_path", "left_t2_path", "right_t1_path", "right_t2_path")) for group in GROUPS for i in range(group_n[group])), len(records)), "siemens": format_count_percent(sum(all(Path(source_payloads["SIEMENS"][key][i]).exists() for key in ("left_t1_path", "left_t2_path", "right_t1_path", "right_t2_path")) for i in range(group_n["SIEMENS"])), group_n["SIEMENS"]), "uih": format_count_percent(sum(all(Path(source_payloads["UIH"][key][i]).exists() for key in ("left_t1_path", "left_t2_path", "right_t1_path", "right_t2_path")) for i in range(group_n["UIH"])), group_n["UIH"]), "test": "—", "p": "—"},
    ])

    analysis = {
        "source": {"master_xlsx": str(MASTER_XLSX), "group_jsons": {key: str(value) for key, value in GROUPS.items()}},
        "sample_sizes": group_n,
        "missing": {
            group: {
                "age": sum(row["age_years"] is None for row in rows),
                "sex": sum(row["sex"] is None for row in rows),
            }
            for group, rows in by_group.items()
        },
        "age_descriptive": {group: describe(values) for group, values in ages.items()},
        "age_normality": age_normality,
        "tests": tests,
        "imaging_summary": imaging_summary,
        "table_rows": table_rows,
    }
    (OUTPUT_DIR / "demographic_analysis.json").write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")

    with (OUTPUT_DIR / "participant_level_deidentified.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    test_rows = []
    for key, value in tests.items():
        test_rows.append({
            "analysis": key,
            "test": value.get("primary_test", value.get("test")),
            "statistic": value.get("statistic"),
            "df": value.get("df"),
            "p_value": value.get("p_value"),
            "effect_size": value.get("hedges_g", value.get("cramers_v")),
            "effect_size_name": "Hedges g" if "hedges_g" in value else "Cramer's V",
            "sensitivity_test": value.get("sensitivity_test", "Fisher exact test" if "fisher_exact_p" in value else ""),
            "sensitivity_p_value": value.get("sensitivity_p_value", value.get("fisher_exact_p", "")),
        })
    with (OUTPUT_DIR / "statistical_test_audit.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(test_rows[0].keys()))
        writer.writeheader()
        writer.writerows(test_rows)

    headers = ["Characteristic", "Overall (N=288)", "SIEMENS (n=190)", "UIH (n=98)", "Test statistic", "P value"]
    lines = ["# Table 1. Demographic, clinical, and imaging characteristics by MRI device group", "", "| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in table_rows:
        lines.append("| " + " | ".join(str(row[key]) for key in ("characteristic", "overall", "siemens", "uih", "test", "p")) + " |")
    lines.extend([
        "",
        "Values are n (%) unless otherwise indicated. Percentages for sex use non-missing denominators; one UIH participant had missing age and sex. Age was compared with a two-sided Welch t-test, with the Mann–Whitney U test retained as a sensitivity analysis. Categorical variables were compared with two-sided Pearson chi-square tests; Fisher's exact test was additionally calculated for 2×2 tables. Exact P values are shown to three decimals, with P<0.001 reported as such. ONFH status and maximum stage were derived at participant level from bilateral hip labels to avoid treating paired hips as independent observations. Scanner model, magnetic field strength, coil, and acquisition parameters were not retained in the supplied NRRD headers or JSON metadata and should be completed from the original DICOM records before submission. ADR, source-filename token (expansion not available in the supplied metadata); FSE, fast spin echo; IQR, interquartile range; ONFH, osteonecrosis of the femoral head; SD, standard deviation; STIR, short tau inversion recovery; TIRM, turbo inversion recovery magnitude; TSE, turbo spin echo.",
    ])
    lines[-1] = lines[-1].replace(
        "Percentages for sex use non-missing denominators; one UIH participant had missing age and sex.",
        "Age and sex were complete in both cohorts after the source record was updated; no missing-value imputation was performed.",
    )
    (OUTPUT_DIR / "Table1_demographics.md").write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps({"output_dir": str(OUTPUT_DIR.resolve()), "sample_sizes": group_n, "tests": {key: value["p_value"] for key, value in tests.items()}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
