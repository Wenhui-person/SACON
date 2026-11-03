from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import pydicom


TAGS = [
    "Manufacturer", "ManufacturerModelName", "MagneticFieldStrength",
    "SoftwareVersions", "InstitutionName", "BodyPartExamined", "PatientPosition",
    "ReceiveCoilName", "TransmitCoilName", "MRAcquisitionType",
    "ScanningSequence", "SequenceVariant", "ScanOptions", "SequenceName",
    "SeriesDescription", "ProtocolName", "RepetitionTime", "EchoTime",
    "InversionTime", "FlipAngle", "SliceThickness", "SpacingBetweenSlices",
    "PixelSpacing", "Rows", "Columns", "SeriesInstanceUID",
]


def normalize(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [normalize(v) for v in value]
    try:
        if value.__class__.__name__ in {"MultiValue"}:
            return [normalize(v) for v in value]
    except Exception:
        pass
    if isinstance(value, (int, float, str)):
        return value
    return str(value)


def read_series_headers(case_dir: Path):
    by_uid = {}
    for path in case_dir.rglob("*"):
        if not path.is_file():
            continue
        try:
            ds = pydicom.dcmread(
                path,
                stop_before_pixels=True,
                specific_tags=TAGS,
                force=True,
            )
        except Exception:
            continue
        uid = str(getattr(ds, "SeriesInstanceUID", ""))
        if not uid or uid in by_uid:
            continue
        rec = {tag: normalize(getattr(ds, tag, None)) for tag in TAGS if tag != "SeriesInstanceUID"}
        desc = " ".join(str(rec.get(k) or "") for k in ("SeriesDescription", "ProtocolName", "SequenceName")).lower()
        if "t1" in desc and ("cor" in desc or "冠" in desc):
            by_uid[uid] = rec
    return list(by_uid.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--cases", type=int, default=12)
    ap.add_argument("--seed", type=int, default=20260910)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    case_dirs = [p for p in args.root.iterdir() if p.is_dir() and p.name.startswith("1.2.826")]
    rng = random.Random(args.seed)
    selected = rng.sample(case_dirs, min(args.cases, len(case_dirs)))
    samples = []
    for idx, case_dir in enumerate(selected, 1):
        for rec in read_series_headers(case_dir):
            rec["sample_case"] = idx
            samples.append(rec)

    grouped = defaultdict(list)
    for rec in samples:
        vendor = str(rec.get("Manufacturer") or "Unknown").strip().upper()
        grouped[vendor].append(rec)

    summary = {"seed": args.seed, "case_count": len(selected), "t1_coronal_series": len(samples), "vendors": {}}
    for vendor, records in grouped.items():
        fields = {}
        for tag in TAGS:
            if tag == "SeriesInstanceUID":
                continue
            vals = [json.dumps(r.get(tag), ensure_ascii=False, sort_keys=True) for r in records]
            fields[tag] = [{"value": json.loads(v), "n": n} for v, n in Counter(vals).most_common()]
        summary["vendors"][vendor] = {"n_series": len(records), "fields": fields}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "vendors"}, ensure_ascii=False))
    for vendor, info in summary["vendors"].items():
        print(vendor, info["n_series"])


if __name__ == "__main__":
    main()
