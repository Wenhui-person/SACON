from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import itertools
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba
from matplotlib.image import AxesImage
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
from sklearn.calibration import calibration_curve

VERSION = "mrnet-subgroups-1"
STRATEGIES = ("zero_shot", "siemens_finetune", "uih_ssl_only")
LABELS = {"zero_shot": "Zero-shot", "siemens_finetune": "SIEMENS fine-tune",
          "uih_ssl_only": "UIH SSL-only", "siemens_internal_oof": "Internal repeated holdout"}
STYLE = {s: (c, m) for s, c, m in zip(
    (*STRATEGIES, "siemens_internal_oof"),
    ("#0072B2", "#D55E00", "#009E73", "#0072B2"), ("o", "s", "^", "o"))}
GROUPS = {"overall": ("Overall",), "sex": ("Female", "Male"),
          "age_group": ("<40", "40–59", "≥60"),
          "disease_pattern": ("Bilateral normal", "Unilateral ONFH", "Bilateral ONFH"),
          "stage_concordance": ("Concordant", "Discordant")}
EARLY = {"bin": ("early_miss_rate",),
         "stage": ("early_to_normal", "early_to_late", "early_recall")}


def csv(frame, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig", na_rep="NA")


def dump(value, path):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def fingerprint(paths, settings):
    sources = []
    for path in sorted(set(map(str, paths))):
        p = Path(path)
        h = hashlib.sha256()
        with p.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                h.update(block)
        sources.append({"path": str(p.resolve()), "sha256": h.hexdigest(), "bytes": p.stat().st_size})
    return {"version": VERSION, "sources": sources, "settings": settings}


def cache_valid(folder, expected):
    try:
        saved = json.loads((folder / "source_audit.json").read_text(encoding="utf-8"))
        if saved["input_fingerprint"] != expected:
            return False
        return saved["prediction_sha256"] == hashlib.sha256((folder / "subgroup_predictions.csv").read_bytes()).hexdigest()
    except (OSError, ValueError, KeyError):
        return False


def seal_cache(folder, expected):
    dump({"input_fingerprint": expected, "prediction_sha256": hashlib.sha256(
        (folder / "subgroup_predictions.csv").read_bytes()).hexdigest()}, folder / "source_audit.json")


def audit_internal_design(api, config, folder):
    rows, patient_rows = [], []
    for task in config.tasks:
        checkpoints = api.discover_fold_checkpoints(getattr(config, f"{task}_checkpoint_dir"), getattr(config, f"{task}_checkpoint_name"), task)
        for fold, checkpoint in api.validate_checkpoints(checkpoints, task):
            path = Path(checkpoint).parent / "fold_data_split.csv"
            table = pd.read_csv(path, dtype={"patient_id": str})
            test = set(table.loc[table.split == "test", "patient_id"])
            train = table.loc[table.split == "train", "patient_id"]
            synthetic = train[train.str.startswith("gen_full_")]
            origins = set(synthetic.str.extract(r"(mri\d+)")[0].dropna())
            exposed = test & origins
            rows.append({"task": task, "fold": fold, "n_test_patients": len(test),
                         "n_synthetic_source_overlap": len(exposed), "split_source": str(path)})
            patient_rows.extend({"task": task, "fold": fold, "patient_id": p, "synthetic_source_in_training": p in exposed} for p in sorted(test))
    csv(pd.DataFrame(rows), folder / "training_overlap_audit.csv")
    csv(pd.DataFrame(patient_rows), folder / "patient_training_overlap_audit.csv")


def saved_indices(api, checkpoint, samples):
    path = Path(checkpoint).parent / "fold_data_split.csv"
    table = pd.read_csv(path, dtype={"patient_id": str})
    if not {"split", "patient_id"} <= set(table):
        raise ValueError(f"Missing saved split columns: {path}")
    test = table[table.split.str.lower() == "test"].copy()
    train = table[table.split.str.lower().isin(["train", "training"])]
    if test.empty or test.patient_id.duplicated().any():
        raise ValueError(f"Empty or duplicate saved test patients: {path}")
    test_base = set(test.patient_id.map(api.get_base_patient_id))
    train_base = set(train.patient_id.map(api.get_base_patient_id))
    if test_base & train_base:
        raise ValueError(f"Train/test patient overlap (including augmentation origins): {path}")
    if test.patient_id.map(api.is_synthetic_sample).any():
        raise ValueError(f"Synthetic test patient: {path}")
    if "sample_type" in test and (test.sample_type != "real").any():
        raise ValueError(f"Non-real test sample: {path}")
    ids = list(map(str, samples["patient_id"]))
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate patient IDs in image manifest")
    lookup = {pid: i for i, pid in enumerate(ids)}
    absent = set(test.patient_id) - set(lookup)
    if absent:
        raise ValueError(f"{len(absent)} saved test patients absent from image manifest: {path}")
    indices = np.array([lookup[p] for p in test.patient_id], dtype=int)
    for column in ("label_left_bin", "label_right_bin", "label_left_ELS", "label_right_ELS"):
        if column not in test or column not in samples:
            raise ValueError(f"Missing label {column}: {path}")
        if not np.array_equal(test[column].to_numpy(int), np.asarray(samples[column])[indices].astype(int)):
            raise ValueError(f"Saved split / image manifest label conflict: {column}, {path}")
    return indices, str(path), test.patient_id.tolist()


def infer_saved_oof(api, task, checkpoints, args, device):
    samples = json.loads(Path(args.data_root).read_text(encoding="utf-8"))
    specs, seen, audit, repeated = [], set(), [], False
    for fold, checkpoint in api.validate_checkpoints(checkpoints, task):
        indices, split, ids = saved_indices(api, checkpoint, samples)
        overlap = len(seen.intersection(ids))
        repeated = repeated or bool(overlap)
        seen.update(ids)
        specs.append((fold, checkpoint, indices, split))
        audit.append({"task": task, "fold": fold, "checkpoint": checkpoint,
                      "split_source": split, "n_patients": len(ids), "n_previously_tested": overlap,
                      "status": "validated"})
    csv(pd.DataFrame(audit), Path(args.out_dir) / f"{task}_split_audit.csv")
    rows = []
    for fold, checkpoint, indices, split in specs:
        print(f"[INFER] {task} fold={fold}, patients={len(indices)}", flush=True)
        dataset, loader = api.make_loader(args.data_root, indices, tuple(args.volume_size), args.batch_size, args.num_workers)
        model, metadata = api.load_model_from_checkpoint(checkpoint, task, device)
        frame = api.infer_predictions(model=model, loader=loader, dataset=dataset,
            task=task, checkpoint_name=Path(checkpoint).stem, device=device,
            num_classes=int(metadata.get("train_cfg", {}).get("num_stages", 3)), threshold=args.threshold)
        frame["fold"], frame["cohort"] = fold, args.cohort
        frame["split_source"], frame["source_file"] = split, checkpoint
        frame["evaluation_mode"] = "internal_saved_split_oof"
        rows.append(frame)
        del model
        if device.type == "cuda":
            api.torch.cuda.empty_cache()
    raw = pd.concat(rows, ignore_index=True)
    csv(raw, Path(args.out_dir) / f"{task}_heldout_fold_predictions.csv")
    if repeated:
        return aggregate_heldout(raw, task, args.threshold)
    return raw


def aggregate_heldout(raw, task, threshold):
    keys = ["cohort", "task", "patient_id", "side"]
    if raw.duplicated(keys + ["fold"]).any():
        raise ValueError("Duplicate held-out prediction within a split")
    rows = []
    probs = ["prob_positive"] if task == "bin" else [f"prob_class_{i}" for i in range(3)]
    for _, group in raw.groupby(keys, sort=False):
        if group.label.nunique() != 1:
            raise ValueError("Label conflict across held-out predictions")
        row = group.iloc[0].to_dict()
        values = group[probs].mean().to_numpy(float)
        row.update(dict(zip(probs, values)))
        row["prediction"] = int(values[0] >= threshold) if task == "bin" else int(values.argmax())
        row["confidence"] = float(max(values[0], 1 - values[0])) if task == "bin" else float(values.max())
        row["is_error"] = row["prediction"] != row["label"]
        row["fold"] = ";".join(map(str, sorted(group.fold.unique())))
        row["heldout_model_count"] = group.fold.nunique()
        for column in ("source_file", "split_source"):
            row[column] = ";".join(group[column].drop_duplicates().astype(str))
        row["evaluation_mode"] = "repeated_holdout_eligible_model_average"
        rows.append(row)
    return pd.DataFrame(rows)


def external_files(root, tasks=("bin", "stage"), strategies=STRATEGIES):
    directories = {"zero_shot": "zero_shot-uih", "siemens_finetune": "siemens_finetune", "uih_ssl_only": "ssl-uih"}
    return [(task, strategy, Path(root) / f"{directories[strategy]}-{task}" / task /
             (f"{task}_zero_shot_ensemble_predictions.csv" if strategy == "zero_shot" else f"{task}_oof_predictions.csv"))
            for strategy in STRATEGIES if strategy in strategies for task in tasks]


def validate_rows(api, frame):
    if frame.empty:
        raise ValueError("No prediction rows to analyze")
    key = ["cohort", "strategy", "task", "patient_id", "side"]
    if frame[key].isna().any().any() or frame.duplicated(key).any():
        raise ValueError("Null or duplicate prediction key")
    if not set(frame.side) <= {"left", "right"}:
        raise ValueError("Unknown hip side")
    if not set(frame.task) <= {"bin", "stage"}:
        raise ValueError("Unknown prediction task")
    for task, group in frame.groupby("task", sort=False):
        api.prediction_columns_ok(group, task)
        allowed = {0, 1} if task == "bin" else {0, 1, 2}
        if not set(group.label) <= allowed or not set(group.prediction) <= allowed:
            raise ValueError(f"Invalid {task} labels/predictions")
        cols = ["prob_positive"] if task == "bin" else [f"prob_class_{i}" for i in range(3)]
        values = group[cols].to_numpy(float)
        if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
            raise ValueError("Non-finite or out-of-range probabilities")
        if task == "stage" and not np.allclose(values.sum(axis=1), 1, atol=1e-3):
            raise ValueError("Stage probabilities do not sum to one")


def pairing_audit(frame, folder):
    issues = []
    for task, data in frame.groupby("task", sort=False):
        strategies = [s for s in STRATEGIES if s in set(data.strategy)]
        if len(strategies) < 2:
            continue
        ref = data[data.strategy == strategies[0]].set_index(["patient_id", "side"])
        for strategy in strategies[1:]:
            other = data[data.strategy == strategy].set_index(["patient_id", "side"])
            for key in ref.index.difference(other.index):
                issues.append((task, strategy, *key, "missing"))
            for key in other.index.difference(ref.index):
                issues.append((task, strategy, *key, "extra"))
            common = ref.index.intersection(other.index)
            for key in common[ref.loc[common, "label"].to_numpy() != other.loc[common, "label"].to_numpy()]:
                issues.append((task, strategy, *key, "label_conflict"))
    csv(pd.DataFrame(issues, columns=["task", "strategy", "patient_id", "side", "issue"]), folder / "pairing_audit.csv")
    if issues:
        raise ValueError(f"External pairing failed; see {folder / 'pairing_audit.csv'}")


def import_external(api, args):
    folder = Path(args.out_dir)
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    for task, strategy, path in external_files(args.external_results_root, args.tasks, args.strategies):
        frame = pd.read_csv(path, dtype={"patient_id": str})
        if frame.empty or set(frame.task) != {task}:
            raise ValueError(f"Empty or wrong task: {path}")
        frame["strategy"], frame["cohort"] = strategy, "UIH_external_cross_protocol"
        frame["source_file"] = str(path)
        frame["evaluation_mode"] = "external_zero_shot_ensemble" if strategy == "zero_shot" else "target_domain_adaptation_oof"
        if strategy != "zero_shot" and ("outer_fold" not in frame or frame.outer_fold.isna().any()):
            raise ValueError(f"Missing OOF provenance: {path}")
        if task == "bin":
            frame["prediction"] = (frame.prob_positive >= args.threshold).astype(int)
        rows.append(frame)
    predictions = pd.concat(rows, ignore_index=True)
    validate_rows(api, predictions)
    pairing_audit(predictions, folder)
    predictions, audit = api.attach_demographics(predictions, args.demographics_xlsx, False)
    predictions = enrich_clinical(predictions, folder)
    csv(predictions, folder / "subgroup_predictions.csv")
    csv(audit, folder / "demographic_audit.csv")
    dump({"sources": [str(p) for _, _, p in external_files(args.external_results_root, args.tasks, args.strategies)],
          "n_patients": predictions.patient_id.nunique(), "n_rows": len(predictions),
          "threshold": args.threshold, "missingness": "per-analysis variable"}, folder / "subgroup_inference_manifest.json")


def enrich_clinical(frame, audit_folder=None):
    frame = frame.drop(columns=["overall", "disease_pattern", "stage_concordance", "stage_gap", "true_stage"], errors="ignore").copy()
    keys = ["cohort", "patient_id", "side"]
    truth = frame[keys + ["task", "label"]].drop_duplicates()
    if truth.duplicated(keys + ["task"]).any():
        raise ValueError("Conflicting labels across strategies")
    stage = truth[truth.task == "stage"][keys + ["label"]].rename(columns={"label": "true_stage"})
    frame = frame.merge(stage, on=keys, how="left", validate="many_to_one")
    bin_truth = truth[truth.task == "bin"][keys + ["label"]]
    st_truth = stage.rename(columns={"true_stage": "label"}).copy()
    st_truth["label"] = (st_truth.label > 0).astype(int)
    check = bin_truth.merge(st_truth, on=keys, suffixes=("_bin", "_stage"))
    conflicts = check[check.label_bin != check.label_stage]
    if audit_folder is not None:
        csv(conflicts, Path(audit_folder) / "cross_task_label_conflicts.csv")
    elif len(conflicts):
        raise ValueError("Binary / stage disease labels disagree")
    conflict_patients = set(zip(conflicts.cohort, conflicts.patient_id))
    conflict_hips = set(map(tuple, conflicts[keys].to_numpy()))
    disease = pd.concat([bin_truth, st_truth]).drop_duplicates(keys, keep="first")
    records = []
    stage_lookup = stage.set_index(keys).true_stage.to_dict()
    for (cohort, pid), group in disease.groupby(["cohort", "patient_id"], sort=False):
        pattern = concordance = None
        gap = np.nan
        if set(group.side) == {"left", "right"} and (cohort, pid) not in conflict_patients:
            pattern = GROUPS["disease_pattern"][int(group.label.sum())]
        values = [stage_lookup.get((cohort, pid, side), np.nan) for side in ("left", "right")]
        if np.isfinite(values).all():
            gap = abs(values[0] - values[1])
            concordance = "Concordant" if gap == 0 else "Discordant"
        records.append({"cohort": cohort, "patient_id": pid, "disease_pattern": pattern,
                        "stage_concordance": concordance, "stage_gap": gap})
    frame = frame.merge(pd.DataFrame(records), on=["cohort", "patient_id"], how="left", validate="many_to_one")
    frame["overall"] = "Overall"
    conflict_mask = np.array([tuple(row) in conflict_hips for row in frame[keys].to_numpy()])
    frame["cross_task_label_conflict"] = conflict_mask
    frame.loc[conflict_mask & (frame.task == "bin"), "true_stage"] = np.nan
    frame["is_error"] = frame.prediction != frame.label
    return frame


def weighted_metrics(frame, task, weights):
    w = np.atleast_2d(weights).astype(float)
    y, pred = frame.label.to_numpy(int), frame.prediction.to_numpy(int)
    k = 2 if task == "bin" else 3
    cm = np.stack([w[:, (y == i) & (pred == j)].sum(axis=1)
                   for i in range(k) for j in range(k)], axis=1).reshape(-1, k, k)
    actual, guessed, total = cm.sum(axis=2), cm.sum(axis=1), cm.sum(axis=(1, 2))
    diag = np.diagonal(cm, axis1=1, axis2=2)
    def div(a, b):
        return np.divide(a, b, out=np.full(np.broadcast_shapes(np.shape(a), np.shape(b)), np.nan), where=np.asarray(b) != 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        recall = div(diag, actual)
        f1 = div(2 * diag, actual + guessed)
        f1 = np.nan_to_num(f1)
        if task == "bin":
            order = np.argsort(frame.prob_positive.to_numpy(float), kind="stable")
            scores = frame.prob_positive.to_numpy(float)[order]
            starts = np.r_[0, np.flatnonzero(np.diff(scores)) + 1] if len(scores) else np.array([], int)
            if len(starts):
                pos = np.add.reduceat(w[:, order] * (y[order] == 1), starts, axis=1)
                neg = np.add.reduceat(w[:, order] * (y[order] == 0), starts, axis=1)
                auc = div((pos * (np.cumsum(neg, axis=1) - .5 * neg)).sum(axis=1), actual[:, 0] * actual[:, 1])
                tp = np.cumsum(pos[:, ::-1], axis=1)
                fp = np.cumsum(neg[:, ::-1], axis=1)
                ap = div(np.nansum(div(tp, tp + fp) * pos[:, ::-1], axis=1), actual[:, 1])
                ap[(actual == 0).any(axis=1)] = np.nan
            else:
                auc = ap = np.full(len(w), np.nan)
            result = {"auroc": auc, "auprc": ap, "sensitivity": recall[:, 1], "specificity": recall[:, 0],
                      "balanced_accuracy": recall.mean(axis=1), "f1": f1[:, 1],
                      "brier": div(w @ ((frame.prob_positive.to_numpy(float) - y) ** 2), total)}
        else:
            distance = (np.arange(3)[:, None] - np.arange(3)[None, :]) ** 2
            observed = (cm * distance).sum(axis=(1, 2))
            expected = div((actual[:, :, None] * guessed[:, None, :] * distance).sum(axis=(1, 2)), total)
            qwk = 1 - div(observed, expected)
            qwk[(actual > 0).sum(axis=1) < 2] = np.nan
            result = {"qwk": qwk, "macro_f1": f1.mean(axis=1),
                      "balanced_accuracy": div(np.nansum(recall, axis=1), (actual > 0).sum(axis=1)),
                      "mae": div(w @ np.abs(y - pred), total),
                      "within_one_stage_accuracy": div(w @ (np.abs(y - pred) <= 1), total),
                      "severe_cross_stage_error_rate": div(w @ (np.abs(y - pred) > 1), total)}
        early = frame.true_stage.to_numpy(float) == 1
        denom = w[:, early].sum(axis=1)
        for name, target in (("early_miss_rate", 0),) if task == "bin" else (("early_to_normal", 0), ("early_to_late", 2), ("early_recall", 1)):
            result[name] = div(w @ (early & (pred == target)), denom)
        for values in result.values():
            values[total == 0] = np.nan
    return result


def cluster_weights(frame, n, rng):
    ids, codes = np.unique(frame.patient_id.astype(str), return_inverse=True)
    if not len(ids):
        return np.zeros((n, 0))
    return rng.multinomial(len(ids), np.full(len(ids), 1 / len(ids)), size=n)[:, codes]


def interval(values):
    valid = np.asarray(values)[np.isfinite(values)]
    if len(valid) < 20:
        return np.nan, np.nan, len(valid)
    return *np.percentile(valid, [2.5, 97.5]), len(valid)


def bootstrap_moments(values):
    valid = np.asarray(values, dtype=float)
    valid = valid[np.isfinite(valid)]
    if not len(valid):
        return np.nan, np.nan
    mean = float(valid.mean())
    sd = float(valid.std(ddof=1)) if len(valid) > 1 else np.nan
    return mean, sd


def centered_bootstrap_p(samples, estimate):
    samples = np.asarray(samples)[np.isfinite(samples)]
    if len(samples) < 20 or not np.isfinite(estimate):
        return np.nan
    return (1 + np.count_nonzero(np.abs(samples - estimate) >= abs(estimate))) / (len(samples) + 1)


def summarize(api, frame, n, seed):
    rng = np.random.default_rng(seed)
    rows, gaps, draws = [], [], {}
    for (cohort, strategy, task), data in frame.groupby(["cohort", "strategy", "task"], sort=False):
        for variable, levels in GROUPS.items():
            for level in levels:
                group = data[data[variable] == level]
                point = weighted_metrics(group, task, np.ones((1, len(group))))
                boot = weighted_metrics(group, task, cluster_weights(group, n, rng))
                draws[(strategy, task, variable, level)] = (point, boot)
                for metric, value in point.items():
                    early_metric = metric in EARLY[task]
                    denominator = group[group.true_stage == 1] if early_metric else group
                    low, high, valid = interval(boot[metric])
                    boot_mean, boot_sd = bootstrap_moments(boot[metric])
                    rows.append({"cohort": cohort, "strategy": strategy, "task": task,
                        "subgroup_variable": variable, "subgroup": level, "metric": metric,
                        "estimate": value[0], "bootstrap_mean": boot_mean, "bootstrap_sd": boot_sd,
                        "ci_low": low, "ci_high": high, "n_bootstrap_valid": valid,
                        "n_patients": denominator.patient_id.nunique(), "n_hips": len(denominator),
                        **{f"n_class_{i}": int((denominator.label == i).sum()) for i in range(3)},
                        "status": "estimable" if np.isfinite(value[0]) else "not_estimable"})
            comparisons = [("Male", "Female")] if variable == "sex" else [("<40", "40–59"), ("≥60", "40–59")] if variable == "age_group" else []
            for left, right in comparisons:
                a, ab = draws[(strategy, task, variable, left)]
                b, bb = draws[(strategy, task, variable, right)]
                for metric in api.BINARY_METRICS if task == "bin" else api.STAGE_METRICS:
                    delta = float(a[metric][0] - b[metric][0])
                    samples = ab[metric] - bb[metric]
                    low, high, valid = interval(samples)
                    boot_mean, boot_sd = bootstrap_moments(samples)
                    gaps.append({"cohort": cohort, "strategy": strategy, "task": task,
                        "subgroup_variable": variable, "comparison": f"{left} vs {right}",
                        "metric": metric, "difference": delta,
                        "bootstrap_mean": boot_mean, "bootstrap_sd": boot_sd,
                        "ci_low": low, "ci_high": high,
                        "n_bootstrap_valid": valid, "p_value": centered_bootstrap_p(samples, delta)})
    gap = pd.DataFrame(gaps)
    if not gap.empty:
        gap["p_holm"] = gap.groupby(["cohort", "strategy", "task"], sort=False).p_value.transform(lambda x: api.holm_adjust(x))
    return pd.DataFrame(rows), gap


def paired_differences(api, frame, n, seed):
    rows, rng = [], np.random.default_rng(seed + 7919)
    for task, data in frame.groupby("task", sort=False):
        strategies = [s for s in STRATEGIES if s in set(data.strategy)]
        for left, right in itertools.combinations(strategies, 2):
            for variable, levels in GROUPS.items():
                for level in levels:
                    a = data[(data.strategy == right) & (data[variable] == level)].sort_values(["patient_id", "side"])
                    b = data[(data.strategy == left) & (data[variable] == level)].sort_values(["patient_id", "side"])
                    if list(zip(a.patient_id, a.side)) != list(zip(b.patient_id, b.side)):
                        raise ValueError("Subgroup pairing mismatch")
                    w = cluster_weights(a, n, rng)
                    ab, bb = weighted_metrics(a, task, w), weighted_metrics(b, task, w)
                    ap, bp = weighted_metrics(a, task, np.ones((1, len(a)))), weighted_metrics(b, task, np.ones((1, len(b))))
                    for metric in ap:
                        delta, samples = ap[metric][0] - bp[metric][0], ab[metric] - bb[metric]
                        low, high, valid = interval(samples)
                        boot_mean, boot_sd = bootstrap_moments(samples)
                        inferential = variable in ("sex", "age_group") and metric not in EARLY[task]
                        denom = a[a.true_stage == 1] if metric in EARLY[task] else a
                        rows.append({"cohort": data.cohort.iloc[0], "task": task,
                            "subgroup_variable": variable, "subgroup": level,
                            "comparison": f"{right} vs {left}", "metric": metric,
                            "difference": delta,
                            "bootstrap_mean": boot_mean, "bootstrap_sd": boot_sd,
                            "ci_low": low, "ci_high": high,
                            "n_paired_patients": denom.patient_id.nunique(), "n_hips": len(denom),
                            "n_bootstrap_valid": valid, "p_value": centered_bootstrap_p(samples, delta) if inferential else np.nan})
    result = pd.DataFrame(rows, columns=["cohort", "task", "subgroup_variable", "subgroup", "comparison",
        "metric", "difference", "bootstrap_mean", "bootstrap_sd", "ci_low", "ci_high",
        "n_paired_patients", "n_hips", "n_bootstrap_valid", "p_value"])
    if not result.empty:
        result["p_holm"] = result.groupby(["task", "metric"], sort=False).p_value.transform(lambda x: api.holm_adjust(x))
    else:
        result["p_holm"] = pd.Series(dtype=float)
    return result


def ordered_strategies(frame):
    return [s for s in ("siemens_internal_oof", *STRATEGIES) if s in set(frame.strategy)]


def gradient_interval(ax, low, high, y):
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return
    mix = np.linspace(0, 1, 192)
    endpoints = np.array([to_rgba("#946368"), to_rgba("#F3D6BD")])
    colors = endpoints[0] * (1 - mix[:, None]) + endpoints[1] * mix[:, None]
    pale = np.array([to_rgba("#EAD9DC"), to_rgba("#F8E9D7")])
    ribbon = np.tile(pale[0] * (1 - mix[:, None]) + pale[1] * mix[:, None], (48, 1, 1))
    ribbon[:, :, 3] *= (np.cos(np.linspace(-np.pi / 2, np.pi / 2, 48)) ** 1.5)[:, None] * .85
    background = AxesImage(ax, extent=(0, 1, y - .115, y + .115),
                           transform=ax.get_yaxis_transform(), origin="lower",
                           interpolation="bilinear", zorder=1)
    background.set_data(ribbon)
    ax.add_image(background)
    ax.plot([low, high], [y, y], color="#70454C", lw=2.8,
            solid_capstyle="round", clip_on=False, zorder=2)
    points = np.column_stack((np.linspace(low, high, len(mix) + 1),
                              np.full(len(mix) + 1, y)))
    segments = np.stack((points[:-1], points[1:]), axis=1)
    ax.add_collection(LineCollection(segments, colors=colors, linewidths=2.0,
                                     capstyle="butt", zorder=2.1, clip_on=False))

def forest(api, metrics, directory, name, panels):
    strategies = ordered_strategies(metrics)
    nrows = max(len(spec[2]) for spec in panels)
    fig, axes = plt.subplots(
        len(panels),
        1,
        figsize=(183 / 25.4, min(170 / 25.4, len(panels) * max(2.6, nrows * 0.44 + 0.95))),
        squeeze=False,
    )
    for panel, (task, metric, levels, title) in enumerate(panels):
        ax = axes[panel, 0]
        for row, (variable, level) in enumerate(levels):
            y = len(levels) - row - 1
            for i, strategy in enumerate(strategies):
                match = metrics[
                    (metrics.task == task)
                    & (metrics.metric == metric)
                    & (metrics.strategy == strategy)
                    & (metrics.subgroup_variable == variable)
                    & (metrics.subgroup == level)
                ]
                if match.empty:
                    continue
                result = match.iloc[0]
                offset = ((len(strategies) - 1) / 2 - i) * 0.32
                _, marker = STYLE[strategy]
                if np.isfinite(result.estimate):
                    if np.isfinite(result.ci_low) and np.isfinite(result.ci_high):
                        gradient_interval(ax, result.ci_low, result.ci_high, y + offset)
                    ax.plot(
                        result.estimate,
                        y + offset,
                        marker,
                        color="#70454C",
                        ms=4.4,
                        markeredgecolor="white",
                        markeredgewidth=0.45,
                        clip_on=False,
                        zorder=3,
                    )
                value = f"{result.estimate:.2f}" if np.isfinite(result.estimate) else "NA"
                bounds = (
                    f" [{result.ci_low:.2f}, {result.ci_high:.2f}]"
                    if np.isfinite(result.ci_low)
                    else ""
                )
                ax.text(
                    1.035,
                    y + offset,
                    f"{value}{bounds}",
                    transform=ax.get_yaxis_transform(),
                    va="center",
                    fontsize=7,
                    color="#222222",
                    clip_on=False,
                )
                ax.text(
                    1.70, y + offset,
                    f"{int(result.n_patients)}/{int(result.n_hips)}",
                    transform=ax.get_yaxis_transform(), va="center",
                    fontsize=7, color="#222222", clip_on=False,
                )
        ax.set_yticks(
            range(len(levels))[::-1],
            [f"{level} years" if variable == "age_group" else level for variable, level in levels],
            fontsize=9,
        )
        ax.tick_params(axis="y", pad=7, length=0)
        ax.spines["left"].set_visible(False)
        ax.set_ylim(-0.65, len(levels) - 0.25)
        displayed = metrics[
            (metrics.task == task) & (metrics.metric == metric)
            & pd.MultiIndex.from_frame(metrics[["subgroup_variable", "subgroup"]]).isin(levels)
        ]
        api.fit_metric_axis(ax, displayed[["estimate", "ci_low", "ci_high"]].to_numpy(), metric)
        ax.set_xlabel(
            api.METRIC_LABELS.get(
                metric, metric.replace("_", " ").capitalize()
            ),
            fontsize=9,
            fontweight="normal",
            labelpad=6,
        )
        ax.set_title(title, fontsize=7, pad=20, loc="left")
        ax.text(
            1.035,
            len(levels) - 0.30,
            "Estimate [95% CI]",
            transform=ax.get_yaxis_transform(),
            fontsize=7,
            clip_on=False,
        )
        ax.text(1.70, len(levels) - 0.30, "Patients/hips",
                transform=ax.get_yaxis_transform(), fontsize=7, clip_on=False)
        api.clean_axis(ax, "x")
        ax.text(
            -0.52,
            1.07,
            chr(97 + panel),
            transform=ax.transAxes,
            fontsize=8,
            fontweight="bold",
            clip_on=False,
        )
    handles = [
        Line2D(
            [], [], color="#946368", marker=STYLE[s][1],
            label=LABELS[s], lw=2.0, markerfacecolor="#70454C", markeredgecolor="white",
            markeredgewidth=0.45
        )
        for s in strategies
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(strategies),
        fontsize=8,
        frameon=False,
    )
    fig.subplots_adjust(left=0.24, right=0.64, bottom=0.10, top=0.91, hspace=0.42)
    api.save_figure(fig, directory / name)


def plot_curves(api, frame, directory):
    strategies = ordered_strategies(frame)
    data = frame[frame.task == "bin"]
    if data.empty:
        return
    width = 89 / 25.4 if len(strategies) == 1 else 183 / 25.4
    fig, axes = plt.subplots(
        3,
        len(strategies),
        figsize=(width, 165 / 25.4),
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    records = []
    levels = [
        ("overall", "Overall"),
        *[(v, level) for v in ("sex", "age_group") for level in GROUPS[v]],
    ]
    for col, strategy in enumerate(strategies):
        cohort = data[data.strategy == strategy]
        for variable, level in levels:
            group = cohort[cohort[variable] == level]
            if group.label.nunique() < 2:
                continue
            color = "#222222" if level == "Overall" else api.GROUP_COLORS[level]
            style = "-" if variable in ("overall", "sex") else "--"
            width_line = 1.45 if level == "Overall" else 0.95
            fpr, tpr, _ = roc_curve(group.label, group.prob_positive)
            precision, recall, _ = precision_recall_curve(
                group.label, group.prob_positive
            )
            curves = [(0, fpr, tpr, "ROC"), (1, recall, precision, "PR")]
            if len(group) >= 20:
                observed, predicted = calibration_curve(
                    group.label,
                    group.prob_positive,
                    n_bins=min(8, max(3, len(group) // 12)),
                    strategy="quantile",
                )
                curves.append((2, predicted, observed, "Calibration"))
            for row, x, y, kind in curves:
                marker = "o" if kind == "Calibration" else None
                axes[row, col].plot(
                    x,
                    y,
                    color=color,
                    ls=style,
                    lw=width_line,
                    marker=marker,
                    ms=3.5,
                    markerfacecolor="white",
                    markeredgewidth=0.75,
                    label=level,
                    zorder=3,
                )
                records.extend(
                    {
                        "strategy": strategy,
                        "subgroup": level,
                        "curve": kind,
                        "x": a,
                        "y": b,
                    }
                    for a, b in zip(x, y)
                )
        for row in (0, 2):
            axes[row, col].plot(
                [0, 1],
                [0, 1],
                linestyle=(0, (3, 2)),
                color="#999999",
                lw=0.8,
                zorder=1,
            )
        axes[1, col].axhline(
            cohort.label.mean(),
            color="#999999",
            linestyle=(0, (3, 2)),
            lw=0.8,
            zorder=1,
        )
        axes[0, col].set_title(
            LABELS[strategy], fontsize=7, pad=10
        )
        axis_specs = (
            ("False-positive rate", "True-positive rate", "ROC"),
            ("Recall", "Precision", "Precision–recall"),
            ("Mean predicted probability", "Observed event fraction", "Calibration"),
        )
        for row, (x_label, y_label, row_title) in enumerate(axis_specs):
            ax = axes[row, col]
            ax.set_xlim(-0.015, 1.015)
            ax.set_ylim(-0.015, 1.025)
            ax.set_xticks(np.linspace(0, 1, 6))
            ax.set_yticks(np.linspace(0, 1, 6))
            ax.set_xlabel(x_label, labelpad=4)
            ax.set_ylabel(
                y_label if col == 0 else "", labelpad=4
            )
            ax.grid(color="#EDF0F2", linewidth=0.4, zorder=0)
            ax.set_box_aspect(1)
            api.clean_axis(ax)
            api.panel_label(ax, chr(97 + row * len(strategies) + col))
            if col == 0:
                ax.text(
                    -0.43,
                    0.5,
                    row_title,
                    gid="subgroup-title",
                    transform=ax.transAxes,
                    rotation=90,
                    va="center",
                    ha="center",
                    fontsize=7,
                    fontweight="bold",
                    color="#555555",
                )
    handles = [
        Line2D(
            [],
            [],
            color="#222222" if level == "Overall" else api.GROUP_COLORS[level],
            ls="-" if variable in ("overall", "sex") else "--",
            marker="o" if level == "Overall" else None,
            markerfacecolor="white",
            label=level,
            lw=1.5,
        )
        for variable, level in levels
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3 if len(strategies) == 1 else 6,
        fontsize=6.5,
        frameon=False,
        bbox_to_anchor=(0.5, 0.005),
        handlelength=2.2,
        columnspacing=1.0,
    )
    fig.subplots_adjust(
        left=0.23 if len(strategies) == 1 else 0.14,
        right=0.985,
        top=0.94,
        bottom=0.12,
        wspace=0.35,
        hspace=0.44,
    )
    api.save_figure(fig, directory / "fig_binary_curves")
    csv(pd.DataFrame(records), directory.parent / "binary_curve_source_data.csv")


def plot_confusions(api, frame, directory):
    strategies = ordered_strategies(frame)
    data = frame[frame.task == "stage"]
    if data.empty:
        return
    levels = [("overall", "Overall"), *[(v, l) for v in ("sex", "age_group") for l in GROUPS[v]]]
    records = []
    for page, start in enumerate(range(0, len(levels), 3)):
        page_levels = levels[start:start + 3]
        width = 89 / 25.4 if len(strategies) == 1 else 183 / 25.4
        fig, axes = plt.subplots(len(page_levels), len(strategies),
                                 figsize=(width, 160 / 25.4), squeeze=False)
        for row, (variable, level) in enumerate(page_levels):
            for col, strategy in enumerate(strategies):
                ax = axes[row, col]
                g = data[(data.strategy == strategy) & (data[variable] == level)]
                cm = confusion_matrix(g.label, g.prediction, labels=[0, 1, 2])
                proportion = np.divide(cm, cm.sum(axis=1, keepdims=True), out=np.full((3, 3), np.nan), where=cm.sum(axis=1, keepdims=True) != 0)
                im = ax.imshow(np.ma.masked_invalid(proportion), cmap="Blues", vmin=0, vmax=1)
                for i, j in itertools.product(range(3), repeat=2):
                    p = proportion[i, j]
                    text = f"{cm[i,j]}\n{p:.0%}" if np.isfinite(p) else "0\nNA"
                    ax.text(j, i, text, ha="center", va="center", fontsize=6.2,
                            linespacing=1.4, color="white" if p > .55 else "#222222")
                    records.append({"strategy": strategy, "subgroup": level, "true": i, "predicted": j, "count": int(cm[i, j]), "proportion": p})
                ax.set_xticks(range(3), api.STAGE_NAMES, fontsize=6)
                ax.set_yticks(range(3), api.STAGE_NAMES, fontsize=6)
                ax.set_xlabel("Predicted stage", fontsize=6.5)
                ax.set_ylabel("True stage" if col == 0 else "", fontsize=6.5)
                ax.set_title(f"{level}\n{g.patient_id.nunique()} patients · {len(g)} hips",
                             fontsize=6.5, pad=6, linespacing=1.5)
                ax.tick_params(length=0, pad=3)
                for spine in ax.spines.values():
                    spine.set_visible(False)
                ax.set_xticks(np.arange(.5, 2.5), minor=True)
                ax.set_yticks(np.arange(.5, 2.5), minor=True)
                ax.grid(which="minor", color="white", linewidth=.7)
                ax.tick_params(which="minor", length=0)
                api.panel_label(ax, chr(97 + (start + row) * len(strategies) + col))
                if row == 0:
                    ax.text(.5, 1.42, LABELS[strategy], transform=ax.transAxes,
                            ha="center", va="bottom", fontsize=7, gid="subgroup-title")
        fig.subplots_adjust(left=.17 if len(strategies) == 1 else .10,
                            right=.86, top=.85, bottom=.09, hspace=.90, wspace=.50)
        cax = fig.add_axes([.91, .27, .012, .40])
        bar = fig.colorbar(im, cax=cax, label="Row proportion", ticks=[0, .5, 1])
        bar.outline.set_visible(False)
        bar.ax.tick_params(length=2, width=.5, labelsize=6)
        suffix = "" if page == 0 else f"_page_{page + 1}"
        api.save_figure(fig, directory / f"fig_stage_confusions{suffix}")
    csv(pd.DataFrame(records), directory.parent / "stage_confusion_source_data.csv")


def plot_differences(api, data, directory, name):
    if data.empty:
        return
    for task in api.TASKS:
        group = data[(data.task == task) & (data.metric == api.PRIMARY_METRIC[task])].copy()
        if group.empty:
            continue
        group["row_label"] = group.apply(lambda r: (str(r.get("subgroup", "")) + " | " +
            str(r["comparison"]).replace("siemens_finetune", "Fine-tune").replace("uih_ssl_only", "SSL-only").replace("zero_shot", "Zero-shot")), axis=1)
        strategies = ordered_strategies(group) if "strategy" in group else []
        if strategies:
            group["row_label"] = group.strategy.map(LABELS) + " | " + group.row_label
        rows_per_figure = len(group) if name == "fig_strategy_differences" else 16
        for page, start in enumerate(range(0, len(group), rows_per_figure)):
            sheet = group.iloc[start:start + rows_per_figure]
            height = max(2.4, .30 * len(sheet) + 1.05)
            fig, ax = plt.subplots(figsize=(183 / 25.4, height))
            for y, row in enumerate(sheet.itertuples()):
                if np.isfinite(row.difference):
                    if np.isfinite(row.ci_low):
                        gradient_interval(ax, row.ci_low, row.ci_high, y)
                    ax.plot(row.difference, y, "o", color="#70454C", ms=4.4,
                            markeredgecolor="white", markeredgewidth=.4)
                value = f"{row.difference:.2f} [{row.ci_low:.2f}, {row.ci_high:.2f}]" if np.isfinite(row.difference) else "NA"
                ax.text(1.04, y, value, transform=ax.get_yaxis_transform(), va="center", fontsize=6)
            ax.axvline(0, ls=(0, (3, 3)), color="#777777", lw=.7)
            ax.set_yticks(range(len(sheet)), sheet.row_label, fontsize=6)
            ax.set_ylim(len(sheet) - .4, -.8)
            ax.tick_params(axis="y", length=0, pad=6)
            ax.spines["left"].set_visible(False)
            ax.set_xlabel(f"Difference in {api.METRIC_LABELS[api.PRIMARY_METRIC[task]]}")
            title = "Paired strategy differences" if "subgroup" in group else "Demographic subgroup differences"
            ax.set_title(title, fontsize=7, loc="left", pad=15)
            ax.text(1.04, 1.02, "Difference [95% CI]", transform=ax.transAxes, fontsize=6)
            api.clean_axis(ax, "x")
            if name == "fig_strategy_differences":
                fig.subplots_adjust(left=.47, right=.79, bottom=.55 / height, top=1 - .50 / height)
            else:
                fig.subplots_adjust(left=.47, right=.79, bottom=.19 if len(sheet) < 6 else .12, top=.86)
            suffix = "" if page == 0 else f"_page_{page + 1}"
            api.save_figure(fig, directory / f"{name}_{task}{suffix}")


def result_report(api, metrics, differences, folder):
    lines = ["# SACON 亚组分析结果解释", "", "本报告由实际预测自动生成；新增临床亚组属于探索性分析。",
             "患者是重采样单位，左右髋共同抽样。CI反映给定模型预测的抽样不确定性，不包含重新训练的变异。",
             "内部最佳模型如使用同一验证折选定，结果不是嵌套交叉验证的无偏估计。",
             "外部 zero-shot 为源域模型集成；其余两种为目标域适配 OOF，不能统称独立零样本外部验证。", ""]
    audit_path = folder / "demographic_audit.csv"
    if audit_path.exists():
        for a in pd.read_csv(audit_path).itertuples():
            if a.n_source_conflict_ids:
                lines.append(f"**人口学敏感性限制**：按用户指定沿用首条完整标签记录。同一ID存在人口学冲突的患者共{a.n_source_conflict_ids}名；性别缺失{a.n_missing_sex}名，年龄缺失{a.n_missing_age}名。冲突患者的亚组归属未经独立消歧，性别/年龄结论仅作探索性描述，不支持稳健公平性结论。")
            else:
                lines.append(f"人口学审计：未发现同一ID记录冲突；性别缺失{a.n_missing_sex}名，年龄缺失{a.n_missing_age}名。")
    conflict_path = folder / "cross_task_label_conflicts.csv"
    if conflict_path.exists():
        count = len(pd.read_csv(conflict_path))
        if count:
            lines.append(f"跨任务真值冲突共{count}髋：保留各任务原始标签，冲突患者不进入患病模式分析，冲突髋不进入二分类早期漏诊分母；分期任务按其自身真值分析。")
    analysis_frame = pd.read_csv(folder / "analysis_predictions.csv")
    if "evaluation_mode" in analysis_frame and analysis_frame.evaluation_mode.eq("repeated_holdout_eligible_model_average").any():
        lines.append(f"**内部评估设计**：五次重复留出，而非互斥五折；仅平均每位患者真实影像未被纳入该次训练的模型预测。本次覆盖{analysis_frame.patient_id.nunique()}名独立患者；没有获得留出预测的患者未纳入。各患者参与平均的模型数可能不同，详见 heldout_model_count。")
    overlap_path = folder / "training_overlap_audit.csv"
    if overlap_path.exists():
        overlap = pd.read_csv(overlap_path)
        lines.append(f"**内部训练信息重叠限制**：{int((overlap.n_synthetic_source_overlap > 0).sum())}个任务-划分中，训练合成样本的源病例ID与测试患者重叠；每次涉及{overlap.n_synthetic_source_overlap.min()}–{overlap.n_synthetic_source_overlap.max()}名患者。源病例条件生成及全数据生成器开发可能引入测试信息。本结果仅描述既有模型，不能作为无泄漏独立验证。需要按患者隔离生成器及分类器训练后重新评估；本次未重新训练模型。")
    for task in api.TASKS:
        metric = api.PRIMARY_METRIC[task]
        for strategy in ordered_strategies(metrics):
            row = metrics[(metrics.task == task) & (metrics.strategy == strategy) & (metrics.metric == metric) & (metrics.subgroup_variable == "overall")]
            if row.empty:
                continue
            r = row.iloc[0]
            lines.append(f"- {LABELS[strategy]}，{task}总体 {api.METRIC_LABELS[metric]}：{r.estimate:.3f}（95% CI {r.ci_low:.3f}–{r.ci_high:.3f}），{r.n_patients}名患者/{r.n_hips}髋。")
            sub = metrics[(metrics.task == task) & (metrics.strategy == strategy) & (metrics.metric == metric) & metrics.subgroup_variable.isin(["sex", "age_group"])].dropna(subset=["estimate"])
            if len(sub):
                low = sub.loc[sub.estimate.idxmin()]
                lines.append(f"  人口学亚组中观察到的最低点估计为 {low['subgroup']}：{low.estimate:.3f}（95% CI {low.ci_low:.3f}–{low.ci_high:.3f}）；此描述不证明组间差异或公平性问题。")
    early = metrics[(metrics.subgroup_variable == "overall") & metrics.metric.isin(sum((list(v) for v in EARLY.values()), []))]
    lines.extend(["", "## 早期病例", ""])
    for r in early.itertuples():
        lines.append(f"- {LABELS[r.strategy]}，{r.metric}：{r.estimate:.1%}（95% CI {r.ci_low:.1%}–{r.ci_high:.1%}），分母 {r.n_patients}名患者/{r.n_hips}个早期髋。")
    lines.extend(["", "## 分期类别错误检查", ""])
    for strategy, data in analysis_frame[analysis_frame.task == "stage"].groupby("strategy", sort=False):
        late = data[data.label == 2]
        correct = int((late.prediction == 2).sum())
        if len(late):
            lines.append(f"- {LABELS[strategy]}：晚期髋召回率 {correct}/{len(late)}={correct/len(late):.1%}（描述性点估计）；晚期预测为早期 {int((late.prediction == 1).sum())}髋，预测为正常 {int((late.prediction == 0).sum())}髋。总体QWK较高不代表所有分期均识别良好。")
    lines.extend(["", "## 临床亚组", ""])
    clinical = metrics[metrics.subgroup_variable.isin(["disease_pattern", "stage_concordance"])]
    for r in clinical.itertuples():
        desired = api.PRIMARY_METRIC[r.task]
        if r.task == "bin" and r.subgroup == "Bilateral normal":
            desired = "specificity"
        elif r.task == "bin" and r.subgroup == "Bilateral ONFH":
            desired = "sensitivity"
        if r.metric == desired:
            lines.append(f"- {LABELS[r.strategy]}，{r.task}，{r.subgroup}：{r.metric}={r.estimate:.3f}（95% CI {r.ci_low:.3f}–{r.ci_high:.3f}），{r.n_patients}名患者/{r.n_hips}髋。")
    lines.append("上述分层仅描述既有模型与对侧参照条件的关系；未进行交互检验，不能证明模型机制或不同亚组间适配收益存在差异。")
    lines.extend(["", "## 解释边界", "", "- 单类别亚组中无法估计的 AUROC/QWK 记为 NA；双侧正常重点查看特异度，双侧 ONFH 重点查看敏感度。",
        "- 未检出差异不等于性能等效；小亚组和宽置信区间不能支持稳定性结论。",
        "- 策略差值方向为比较名称前者减后者；AUROC/QWK越高越好，漏诊率/MAE/Brier越低越好。",
        "- 既有人口学差异使用零假设中心化 bootstrap 近似双侧检验；有限重复次数校正避免 p=0。",
        "- 人口学组间差异在 cohort×strategy×task 内进行 Holm 校正；外部策略差异在 task×metric 内校正性别/年龄比较。新增临床亚组及总体策略差异仅报告差值和CI。",
        "- 每项分析分母单独计算；人口学缺失不排除其他可用分析。连续年龄回归仅使用完整病例；不收敛或不可识别的模型不解释。", "",
        "## 后续研究建议（本次未实现）", "",
        "1. 策略×亚组交互，直接检验适配收益的异质性。",
        "2. 校准截距、斜率及预先确定临床阈值后的决策曲线。",
        "3. 事件数足够时研究连续年龄非线性及性别×年龄交叉亚组。",
        "4. 元数据齐备后研究病因、病灶范围、图像质量和扫描参数；区分设备与队列混杂。",
        "5. 人口学缺失及各折稳定性的敏感性分析。", "",
        "这些为本课题建议，不是期刊统一必做清单。", "",
        "参考：[Nature figure guide](https://research-figure-guide.nature.com/)，"
        "[npj Digital Medicine 投稿指南](https://www.nature.com/npjdigitalmed/for-authors-and-referees/submission-guidelines)，"
        "[亚组薄弱性能研究](https://www.nature.com/articles/s41746-024-01275-6)。"])
    if not differences.empty:
        lines.extend(["", "## 外部总体策略差异", ""])
        for r in differences[(differences.subgroup_variable == "overall") & differences.metric.isin(api.PRIMARY_METRIC.values())].itertuples():
            lines.append(f"- {r.task}，{r.comparison}，Δ{r.metric}={r.difference:.3f}（95% CI {r.ci_low:.3f}–{r.ci_high:.3f}）。")
    (folder / "results_interpretation.md").write_text("\n".join(lines).replace("nan", "NA"), encoding="utf-8")


def analyze_cohort(api, predictions, args, folder, sources):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    if args.n_bootstrap < 20:
        raise ValueError("At least 20 bootstrap replicates are required; publication default is 2000")
    dump({"status": "running", "n_bootstrap": args.n_bootstrap, "seed": args.seed}, folder / "subgroup_analysis_manifest.json")
    api.set_publication_style()
    api.STRATEGY_LABELS["siemens_internal_oof"] = "Internal repeated holdout"
    frame = api.standardize_strategy_column(predictions)
    frame["age"] = pd.to_numeric(frame.age, errors="coerce")
    frame["age_group"] = frame.age.map(api.assign_age_group)
    validate_rows(api, frame)
    pairing_audit(frame, folder)
    frame = enrich_clinical(frame, folder)
    order = {s: i for i, s in enumerate(("siemens_internal_oof", *STRATEGIES))}
    frame = frame.assign(_order=frame.strategy.map(order)).sort_values(["_order", "task", "patient_id", "side"]).drop(columns="_order")
    print(f"[ANALYZE] {folder.name}: patients={frame.patient_id.nunique()}, rows={len(frame)}, bootstrap={args.n_bootstrap}", flush=True)
    metrics, gaps = summarize(api, frame, args.n_bootstrap, args.seed)
    differences = paired_differences(api, frame, args.n_bootstrap, args.seed)
    csv(frame, folder / "analysis_predictions.csv")
    csv(metrics, folder / "subgroup_metrics.csv")
    csv(gaps, folder / "subgroup_metric_gaps.csv")
    csv(differences, folder / "uih_transfer_strategy_differences.csv")
    denominators = metrics.drop(columns=["estimate", "ci_low", "ci_high", "n_bootstrap_valid", "status"]).drop_duplicates()
    csv(denominators, folder / "analysis_denominators.csv")
    composition = frame.drop_duplicates(["cohort", "patient_id"])
    csv(composition.groupby(["disease_pattern", "stage_concordance", "stage_gap"], dropna=False).size().reset_index(name="n_patients"), folder / "clinical_composition.csv")
    regression, fits = api.regression_summary(frame)
    fits = {key: fit for key, fit in fits.items() if fit.converged}
    csv(regression, folder / "continuous_age_regression.csv")
    main, supplement = folder / "figures", folder / "figures" / "supplementary"
    main.mkdir(exist_ok=True)
    supplement.mkdir(exist_ok=True)
    demographic = [(v, l) for v in ("overall", "sex", "age_group") for l in GROUPS[v]]
    clinical = [(v, l) for v in ("disease_pattern", "stage_concordance") for l in GROUPS[v]]
    tasks = [t for t in api.TASKS if t in set(frame.task)]
    forest(api, metrics, main, "fig_demographic_performance", [(t, api.PRIMARY_METRIC[t], demographic, "Binary classification" if t == "bin" else "Stage classification") for t in tasks])
    forest(api, metrics, main, "fig_clinical_performance", [(t, api.PRIMARY_METRIC[t], clinical, "Binary classification" if t == "bin" else "Stage classification") for t in tasks])
    for task in tasks:
        for metric in EARLY[task]:
            forest(api, metrics, main, f"fig_{task}_{metric}", [(task, metric, demographic + clinical, metric.replace("_", " ").capitalize())])
    if "bin" in tasks:
        forest(api, metrics, main, "fig_clinical_detection", [("bin", m, clinical, m.capitalize()) for m in ("sensitivity", "specificity")])
    for task in tasks:
        for metric in api.BINARY_METRICS if task == "bin" else api.STAGE_METRICS:
            if metric == api.PRIMARY_METRIC[task]:
                continue
            forest(api, metrics, supplement, f"fig_{task}_{metric}", [(task, metric, demographic + clinical, api.METRIC_LABELS[metric])])
    api.plot_continuous_age_effect(frame, fits, supplement)
    plot_differences(api, differences, supplement, "fig_strategy_differences")
    plot_differences(api, gaps, supplement, "fig_demographic_gaps")
    plot_curves(api, frame, supplement)
    plot_confusions(api, frame, supplement)
    result_report(api, metrics, differences, folder)
    (folder / "figure_captions.md").write_text(
        "# Figure captions\n\nAll forest plots show estimates and patient-cluster bootstrap percentile 95% confidence intervals; "
        f"{args.n_bootstrap:,} resamples, seed {args.seed}. Text reports patients/hips for each estimand. "
        "Colours and shapes identify evaluation strategies. NA denotes a non-estimable metric; missing demographic data are excluded only from the corresponding analysis.\n\n"
        "Demographic performance: overall, sex and age strata; binary AUROC and three-class quadratic weighted kappa.\n\n"
        "Clinical performance: true bilateral disease pattern and stage concordance. Clinical detection: sensitivity and specificity; single-class strata cannot support AUROC.\n\n"
        "Early errors: denominators include only true Early hips; binary false negatives, Early-to-Normal, Early-to-Late and Early recall are distinct endpoints.\n\n"
        "Supplementary curves: rows show ROC, precision–recall and quantile-binned calibration; columns show strategies on common axes. PR baseline is the overall cohort prevalence, not each subgroup prevalence. Calibration is shown for at least 20 hips.\n\n"
        "Confusion matrices: true-label row proportions on a shared 0–1 colour scale, with counts; absent true classes are NA.\n\n"
        "Zero-shot uses source-domain model probability ensembling. Fine-tuning and SSL-only retraining use target-domain OOF predictions and are not independent zero-shot validation. "
        "Internal results may include model-selection optimism if the reported fold was also used to select the best checkpoint. "
        "Repeated holdout results average only eligible held-out model predictions per patient, not five mutually exclusive test folds. "
        "Internal training includes synthetic derivatives of some test patients; see training_overlap_audit.csv. Real-image holdout does not imply source-patient independence. "
        "Continuous-age curves use conventional cluster-robust logistic fits where identifiable; "
        "a dagger denotes a ridge-stabilized descriptive curve used after separation and excluded from inferential tables. "
        "Intervals condition on fitted models and do not measure retraining variability. "
        "Demographic conflicts retain the first fully labelled source row by user specification; see demographic_audit.csv. "
        "These subgroup assignments require independent verification before robust demographic conclusions. "
        "Cross-task label conflicts retain task-specific truth, exclude the patient from disease-pattern strata and exclude the conflicting hip from binary early-miss analysis.\n", encoding="utf-8")
    dump({"version": VERSION, "prediction_csvs": list(sources), "n_bootstrap": args.n_bootstrap,
          "seed": args.seed, "bootstrap_unit": "patient", "n_rows": len(frame),
          "n_patients": frame.patient_id.nunique(), "strategy_order": ordered_strategies(frame),
          "missingness": "per variable", "new_analyses": "exploratory; no interaction tests",
          "status": "completed"}, folder / "subgroup_analysis_manifest.json")
    print(f"[OK] Analysis, figures and interpretation: {folder}", flush=True)



def _paper_number(value):
    if pd.isna(value):
        return "NA"
    return f"{float(value):.3f}"


def _add_paper_columns(frame, point_column):
    result = frame.copy()
    if {"bootstrap_mean", "bootstrap_sd"} <= set(result.columns):
        result["paper_mean_sd"] = result.apply(
            lambda row: (
                f"{_paper_number(row['bootstrap_mean'])} ± "
                f"{_paper_number(row['bootstrap_sd'])}"
                if pd.notna(row["bootstrap_mean"])
                else "NA"
            ),
            axis=1,
        )
    if {point_column, "ci_low", "ci_high"} <= set(result.columns):
        result["paper_estimate_95ci"] = result.apply(
            lambda row: (
                f"{_paper_number(row[point_column])} "
                f"({_paper_number(row['ci_low'])} to {_paper_number(row['ci_high'])})"
                if pd.notna(row[point_column])
                else "NA"
            ),
            axis=1,
        )
    return result


def export_excel_summary(output_root):
    output_root = Path(output_root)
    metric_frames, gap_frames, difference_frames = [], [], []
    regression_frames, denominator_frames = [], []
    for cohort in ("internal", "external"):
        folder = output_root / cohort
        sources = (
            ("subgroup_metrics.csv", metric_frames),
            ("subgroup_metric_gaps.csv", gap_frames),
            ("uih_transfer_strategy_differences.csv", difference_frames),
            ("continuous_age_regression.csv", regression_frames),
            ("analysis_denominators.csv", denominator_frames),
        )
        for filename, collection in sources:
            source = folder / filename
            if source.exists() and source.stat().st_size:
                table = pd.read_csv(source)
                if table.empty:
                    continue
                table.insert(0, "dataset", cohort)
                collection.append(table)
    if not metric_frames:
        return None

    metrics = _add_paper_columns(
        pd.concat(metric_frames, ignore_index=True), "estimate"
    )
    preferred = [
        "dataset", "cohort", "strategy", "task", "subgroup_variable",
        "subgroup", "metric", "n_patients", "n_hips", "n_class_0",
        "n_class_1", "n_class_2", "estimate", "bootstrap_mean",
        "bootstrap_sd", "ci_low", "ci_high", "paper_mean_sd",
        "paper_estimate_95ci", "n_bootstrap_valid", "status",
    ]
    metrics = metrics[
        [column for column in preferred if column in metrics.columns]
        + [column for column in metrics.columns if column not in preferred]
    ]

    sheets = {"All metrics": metrics}
    if gap_frames:
        sheets["Demographic gaps"] = _add_paper_columns(
            pd.concat(gap_frames, ignore_index=True), "difference"
        )
    if difference_frames:
        differences = pd.concat(difference_frames, ignore_index=True)
        if not differences.empty:
            sheets["Strategy differences"] = _add_paper_columns(
                differences, "difference"
            )
    if regression_frames:
        sheets["Age regression"] = pd.concat(
            regression_frames, ignore_index=True
        )
    if denominator_frames:
        sheets["Denominators"] = pd.concat(
            denominator_frames, ignore_index=True
        )

    definitions = pd.DataFrame(
        [
            ("Workbook", "SACON subgroup statistics for manuscript tables."),
            (
                "Point estimate",
                "Metric calculated once from the complete eligible evaluation sample.",
            ),
            (
                "Bootstrap mean",
                "Mean across finite patient-cluster bootstrap replicates.",
            ),
            (
                "Bootstrap SD",
                "Sample standard deviation across finite patient-cluster bootstrap replicates (ddof=1).",
            ),
            (
                "95% CI",
                "2.5th and 97.5th percentiles of patient-cluster bootstrap replicates.",
            ),
            (
                "paper_mean_sd",
                "Copy-ready bootstrap mean +/- SD rounded to three decimals; numeric source columns are retained.",
            ),
            (
                "paper_estimate_95ci",
                "Copy-ready point estimate and bootstrap 95% CI rounded to three decimals.",
            ),
            (
                "Internal evaluation",
                "Repeated holdout; eligible held-out model predictions are averaged per patient.",
            ),
            (
                "External evaluation",
                "Zero-shot is external validation; fine-tune and SSL-only are target-domain OOF evaluations.",
            ),
            (
                "Missing values",
                "NA denotes a non-estimable metric or unavailable inferential quantity.",
            ),
            (
                "Age regression",
                "Conventional identifiable models only; ridge-stabilized descriptive curves are excluded.",
            ),
        ],
        columns=["Field", "Definition"],
    )
    sheets["Definitions"] = definitions

    output_path = output_root / "subgroup_metrics_mean_sd.xlsx"
    temporary_path = output_root / "subgroup_metrics_mean_sd.tmp.xlsx"
    try:
        from openpyxl.styles import Alignment, Font, PatternFill
        from openpyxl.utils import get_column_letter
    except ImportError as exc:
        raise RuntimeError(
            "Excel export requires openpyxl, which is also used to read the demographic workbooks."
        ) from exc

    with pd.ExcelWriter(temporary_path, engine="openpyxl") as writer:
        for sheet_name, table in sheets.items():
            table.to_excel(writer, sheet_name=sheet_name, index=False)
        workbook = writer.book
        for sheet_index, (sheet_name, table) in enumerate(sheets.items()):
            worksheet = workbook[sheet_name]
            worksheet.sheet_view.showGridLines = False
            worksheet.freeze_panes = "A2"
            worksheet.auto_filter.ref = worksheet.dimensions
            worksheet.sheet_properties.tabColor = (
                "1F4E78" if sheet_index == 0 else "9FBAD0"
            )
            worksheet.row_dimensions[1].height = 28
            for cell in worksheet[1]:
                cell.fill = PatternFill("solid", fgColor="1F4E78")
                cell.font = Font(name="Arial", size=10, bold=True, color="FFFFFF")
                cell.alignment = Alignment(
                    horizontal="center", vertical="center", wrap_text=True
                )
            for row_index in range(2, worksheet.max_row + 1):
                fill = (
                    PatternFill("solid", fgColor="F3F7FA")
                    if row_index % 2 == 0
                    else PatternFill(fill_type=None)
                )
                for cell in worksheet[row_index]:
                    cell.font = Font(name="Arial", size=10, color="222222")
                    cell.alignment = Alignment(vertical="center")
                    cell.fill = fill
                    if isinstance(cell.value, float):
                        cell.number_format = "0.0000"
                    elif isinstance(cell.value, int):
                        cell.number_format = "#,##0"
            for column_index, column_name in enumerate(table.columns, start=1):
                values = [str(column_name)]
                values.extend(
                    "" if pd.isna(value) else str(value)
                    for value in table[column_name].head(1000)
                )
                width = min(34, max(10, max(map(len, values)) + 2))
                worksheet.column_dimensions[
                    get_column_letter(column_index)
                ].width = width
    temporary_path.replace(output_path)
    print(f"[OK] Excel metric summary: {output_path}", flush=True)
    return output_path

def direct_workflow(api, config, steps=("inference", "analysis"), cohorts=("internal", "external")):
    settings = {"tasks": list(config.tasks), "threshold": config.threshold,
                "volume_size": list(config.volume_size), "complete_case": False,
                "demographic_selection": "first_complete_label_row_user_authorized",
                "internal_aggregation": "eligible_heldout_models_only"}
    if not steps:
        raise ValueError("Enable at least one stage: inference or analysis")
    for cohort in cohorts:
        folder = config.output_root / cohort
        folder.mkdir(parents=True, exist_ok=True)
        if cohort == "internal":
            audit_internal_design(api, config, folder)
            paths = [config.siemens_data_root, config.siemens_demographics_xlsx]
            for task in config.tasks:
                checkpoints = api.discover_fold_checkpoints(getattr(config, f"{task}_checkpoint_dir"), getattr(config, f"{task}_checkpoint_name"), task)
                if len(checkpoints) != 5:
                    raise ValueError(f"Expected five saved {task} checkpoints, found {len(checkpoints)}")
                paths += [Path(p) for p in checkpoints]
                paths += [Path(p).parent / "fold_data_split.csv" for p in checkpoints]
        else:
            paths = [config.uih_demographics_xlsx, *[p for _, _, p in external_files(config.external_results_root, config.tasks, config.uih_transfer_strategies)]]
        print(f"[AUDIT] Hashing {cohort} input sources", flush=True)
        expected = fingerprint(paths, settings)
        valid = config.reuse_existing_predictions and cache_valid(folder, expected)
        if "inference" in steps:
            if valid:
                print(f"[CACHE] Verified {cohort} predictions", flush=True)
            elif cohort == "internal":
                args = api._build_direct_inference_args(config, "SIEMENS_internal", config.siemens_data_root, config.siemens_demographics_xlsx, folder)
                args.complete_case = False
                api.run_inference_command(args, external=False)
                frame = enrich_clinical(pd.read_csv(folder / "subgroup_predictions.csv", dtype={"patient_id": str}), folder)
                validate_rows(api, frame)
                csv(frame, folder / "subgroup_predictions.csv")
                seal_cache(folder, expected)
            else:
                import_external(api, SimpleNamespace(external_results_root=config.external_results_root,
                    tasks=config.tasks, strategies=config.uih_transfer_strategies, out_dir=folder,
                    demographics_xlsx=str(config.uih_demographics_xlsx), threshold=config.threshold))
                seal_cache(folder, expected)
        elif not valid:
            raise ValueError(f"Missing, modified or incompatible {cohort} predictions. Run workflow --steps inference first.")
    if "analysis" in steps:
        for cohort in cohorts:
            folder = config.output_root / cohort
            path = folder / "subgroup_predictions.csv"
            analyze_cohort(api, pd.read_csv(path, dtype={"patient_id": str}),
                SimpleNamespace(n_bootstrap=config.n_bootstrap, seed=config.seed), folder, [str(path)])
        export_excel_summary(config.output_root)


def workflow_command(api, args):
    from dataclasses import replace
    config = replace(api.DIRECT_RUN_CONFIG, n_bootstrap=args.n_bootstrap,
                     output_root=Path(args.out_dir), external_results_root=Path(args.external_results_root))
    steps = ("inference", "analysis") if args.steps == "all" else (args.steps,)
    direct_workflow(api, config, steps, tuple(args.cohorts))
