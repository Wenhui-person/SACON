from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import roc_auc_score, cohen_kappa_score


def verify(root):
    summary = {}
    for cohort in ("internal", "external"):
        folder = root / cohort
        manifest = json.loads((folder / "subgroup_analysis_manifest.json").read_text(encoding="utf-8"))
        assert manifest["status"] == "completed"
        assert manifest["n_bootstrap"] == 2000
        source = pd.read_csv(folder / "subgroup_predictions.csv")
        analysis = pd.read_csv(folder / "analysis_predictions.csv")
        assert not analysis.duplicated(["cohort", "strategy", "task", "patient_id", "side"]).any()
        assert len(analysis) == len(source)
        cache = json.loads((folder / "source_audit.json").read_text(encoding="utf-8"))
        assert hashlib.sha256((folder / "subgroup_predictions.csv").read_bytes()).hexdigest() == cache["prediction_sha256"]
        metrics = pd.read_csv(folder / "subgroup_metrics.csv")
        regression = pd.read_csv(folder / "continuous_age_regression.csv")
        for row in regression.itertuples():
            data = analysis[(analysis.strategy == row.strategy) & (analysis.task == row.task)].dropna(subset=["sex", "age", "label", "side", "patient_id"])
            assert row.n_patients == data.patient_id.nunique()
            assert row.n_hips == len(data)
        for row in metrics.itertuples():
            data = analysis[(analysis.strategy == row.strategy) & (analysis.task == row.task)]
            data = data[data[row.subgroup_variable] == row.subgroup]
            if row.metric.startswith("early_"):
                data = data[data.true_stage == 1]
                target = 0 if row.metric in ("early_miss_rate", "early_to_normal") else 1 if row.metric == "early_recall" else 2
                expected = (data.prediction == target).mean() if len(data) else np.nan
                assert np.isclose(row.estimate, expected, equal_nan=True)
            assert row.n_hips == len(data)
            assert row.n_patients == data.patient_id.nunique()
            for label in range(3):
                assert getattr(row, f"n_class_{label}") == (data.label == label).sum()
            if row.metric in ("auroc", "qwk"):
                expected = np.nan
                if data.label.nunique() >= 2:
                    expected = roc_auc_score(data.label, data.prob_positive) if row.metric == "auroc" else cohen_kappa_score(data.label, data.prediction, labels=[0, 1, 2], weights="quadratic")
                assert np.isclose(row.estimate, expected, equal_nan=True)
        figures = list((folder / "figures").rglob("*.png"))
        for png in figures:
            assert png.with_suffix(".pdf").is_file()
            with Image.open(png) as im:
                assert min(im.size) > 600
                assert all(abs(dpi - 600) < .1 for dpi in im.info["dpi"])
                im.verify()
        if cohort == "internal":
            assert analysis.patient_id.nunique() == 155
            for task in ("bin", "stage"):
                raw = pd.read_csv(folder / f"{task}_heldout_fold_predictions.csv")
                cols = ["prob_positive"] if task == "bin" else [f"prob_class_{i}" for i in range(3)]
                expected = raw.groupby(["patient_id", "side"])[cols].mean().sort_index()
                got = source[source.task == task].set_index(["patient_id", "side"])[cols].sort_index()
                np.testing.assert_allclose(got.to_numpy(), expected.to_numpy())
        else:
            assert analysis.patient_id.nunique() == 98
            assert manifest["strategy_order"] == ["zero_shot", "siemens_finetune", "uih_ssl_only"]
            assert pd.read_csv(folder / "pairing_audit.csv").empty
        assert (folder / "results_interpretation.md").stat().st_size > 100
        summary[cohort] = {"patients": analysis.patient_id.nunique(), "prediction_rows": len(analysis),
                           "metric_rows_checked": len(metrics), "pdf_png_pairs": len(figures), "status": "passed"}
    (root / "verification_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(verify(Path(__file__).resolve().parents[1] / "outputs" / "subgroups_mrnet_ssl"), indent=2))
