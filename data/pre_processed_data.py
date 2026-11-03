

import json
import os
import re
import argparse
from glob import glob
from pathlib import Path

import pandas as pd


def normalize_id(value):
    if pd.isna(value):
        return None

    if isinstance(value, float) and value.is_integer():
        return str(int(value))

    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def json_value(value):
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def find_id_column(df):
    candidates = ["ID", "id", "case_id", "Case ID", "case id", "patient_id", "Patient ID"]
    for column in candidates:
        if column in df.columns:
            return column
    raise KeyError(f"Cannot find ID column in xlsx. Columns: {list(df.columns)}")


def base_patient_id(patient_id):
    return re.sub(r"_aug_\d+$", "", str(patient_id))


def is_augmented_id(patient_id):
    return re.search(r"_aug_\d+$", str(patient_id)) is not None


def collect_flat_paths(image_dir, series_name, exclude_aug=False):
    image_dir = Path(image_dir)
    pattern = str(image_dir / f"{series_name}_*.nrrd")

    paths = {}
    prefix = f"{series_name}_"
    for path in glob(pattern):
        file_name = os.path.basename(path)
        if not file_name.startswith(prefix) or not file_name.endswith(".nrrd"):
            continue
        patient_id = file_name[len(prefix):-len(".nrrd")]
        if exclude_aug and is_augmented_id(patient_id):
            continue
        paths[patient_id] = path
    return paths


def collect_side_paths(data_dir, series_name, side, exclude_aug=False):
    side_lower = side.lower()
    image_dir = Path(data_dir) / ("T1" if series_name.startswith("t1") else "T2") / side
    pattern = str(image_dir / f"{series_name}_{side_lower}_*.nrrd")

    paths = {}
    prefix = f"{series_name}_{side_lower}_"
    for path in glob(pattern):
        file_name = os.path.basename(path)
        if not file_name.startswith(prefix) or not file_name.endswith(".nrrd"):
            continue
        patient_id = file_name[len(prefix):-len(".nrrd")]
        if exclude_aug and is_augmented_id(patient_id):
            continue
        paths[patient_id] = path
    return paths


def collect_image_manifest(data_dir, t1_name, t2_name, exclude_aug=False):
    flat_dir = Path(data_dir) / "T1_and_T2"
    if not flat_dir.exists():
        raise FileNotFoundError(f"T1_and_T2 folder not found: {flat_dir}")

    t1_paths = collect_flat_paths(flat_dir, t1_name, exclude_aug=exclude_aug)
    t2_paths = collect_flat_paths(flat_dir, t2_name, exclude_aug=exclude_aug)

    left_t1_paths = collect_side_paths(data_dir, t1_name, "Left", exclude_aug=exclude_aug)
    left_t2_paths = collect_side_paths(data_dir, t2_name, "Left", exclude_aug=exclude_aug)
    right_t1_paths = collect_side_paths(data_dir, t1_name, "Right", exclude_aug=exclude_aug)
    right_t2_paths = collect_side_paths(data_dir, t2_name, "Right", exclude_aug=exclude_aug)

    file_ids = sorted(
        set(t1_paths)
        & set(t2_paths)
        & set(left_t1_paths)
        & set(left_t2_paths)
        & set(right_t1_paths)
        & set(right_t2_paths)
    )
    return {
        patient_id: {
            "left_t1_path": left_t1_paths[patient_id],
            "left_t2_path": left_t2_paths[patient_id],
            "right_t1_path": right_t1_paths[patient_id],
            "right_t2_path": right_t2_paths[patient_id],
        }
        for patient_id in file_ids
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Build ONFH json manifest from nrrd files and xlsx labels.")
    parser.add_argument("--data_dir", type=str, default=r"E:\3D_MedicalImageProcessing_for_Training\ONFH_MRI_SIEMENS")
    parser.add_argument("--t1_name", type=str, default="t1_tse_cor")
    parser.add_argument("--t2_name", type=str, default="t2_tirm_cor")
    parser.add_argument("--xlsx_path", type=str, default=r"E:\3D_MedicalImageProcessing_for_Training\ONFH_MRI_SIEMENS\info_details.xlsx")
    parser.add_argument("--output_json", type=str, default=r"E:\3D_MedicalImageProcessing_for_Training\ONFH_MRI_SIEMENS\siemens_onfh_data.json")
    parser.add_argument(
        "--exclude_aug",
        action="store_true",
        default=True,
        help="Ignore nrrd files whose patient ID ends with _aug_###. Default keeps previous behavior.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = args.data_dir
    t1_name = args.t1_name
    t2_name = args.t2_name

    xlsx_path = args.xlsx_path or os.path.join(data_dir, "info_details.xlsx")
    output_json_path = args.output_json
    if output_json_path is None:
        output_json_name = "siemens_onfh_data_original_only.json" if args.exclude_aug else "siemens_onfh_data_with_aug.json"
        output_json_path = os.path.join(data_dir, output_json_name)

    paths_by_id = collect_image_manifest(data_dir, t1_name, t2_name, exclude_aug=args.exclude_aug)

    df = pd.read_excel(xlsx_path, header=0)
    id_column = find_id_column(df)
    df["__id__"] = df[id_column].apply(normalize_id)

    required_columns = [
        "Left Label",
        "Left Stage",
        "Right Label",
        "Right Stage",
        "Left ELS",
        "Right ELS",
    ]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise KeyError(f"Missing columns in {xlsx_path}: {missing_columns}")

    label_table = df.dropna(subset=["__id__"] + required_columns).copy()
    for column in required_columns:
        label_table[column] = pd.to_numeric(label_table[column], errors="coerce")
    label_table = label_table.dropna(subset=required_columns)
    for column in required_columns:
        label_table[column] = label_table[column].astype(int)
    label_table = label_table.drop_duplicates("__id__").set_index("__id__")

    samples = {
        "patient_id": [],
        "left_t1_path": [],
        "left_t2_path": [],
        "right_t1_path": [],
        "right_t2_path": [],
        "label_left_bin": [],
        "label_left_stage": [],
        "label_right_bin": [],
        "label_right_stage": [],
        "label_left_ELS": [],
        "label_right_ELS": [],
    }

    missing_label_ids = []
    for patient_id, paths in paths_by_id.items():
        label_id = patient_id if patient_id in label_table.index else base_patient_id(patient_id)
        if label_id not in label_table.index:
            missing_label_ids.append(patient_id)
            continue

        row = label_table.loc[label_id]
        samples["patient_id"].append(patient_id)
        samples["left_t1_path"].append(paths["left_t1_path"])
        samples["left_t2_path"].append(paths["left_t2_path"])
        samples["right_t1_path"].append(paths["right_t1_path"])
        samples["right_t2_path"].append(paths["right_t2_path"])
        samples["label_left_bin"].append(json_value(row["Left Label"]))
        samples["label_left_stage"].append(json_value(row["Left Stage"]))
        samples["label_right_bin"].append(json_value(row["Right Label"]))
        samples["label_right_stage"].append(json_value(row["Right Stage"]))
        samples["label_left_ELS"].append(json_value(row["Left ELS"]))
        samples["label_right_ELS"].append(json_value(row["Right ELS"]))

    print(f"exclude_aug: {args.exclude_aug}")
    print(f"Complete file ids: {len(paths_by_id)}")
    print(f"Samples written: {len(samples['patient_id'])}")
    if missing_label_ids:
        print(f"Skipped ids without xlsx labels: {missing_label_ids}")

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=4, ensure_ascii=False)

    print(f"Saved manifest: {output_json_path}")


if __name__ == '__main__':
    main()
