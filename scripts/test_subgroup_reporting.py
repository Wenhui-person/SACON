import json
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score, average_precision_score, cohen_kappa_score, f1_score
import importlib.util
from pathlib import Path
spec = importlib.util.spec_from_file_location("subgroup_reporting", Path(__file__).resolve().parents[1] / "utils" / "subgroup_reporting.py")
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)


def fixture():
    rows = []
    for strategy in s.STRATEGIES:
        for pid, labels in enumerate(((0, 0), (0, 1), (1, 2), (2, 2))):
            for side, label in zip(("left", "right"), labels):
                for task in ("bin", "stage"):
                    rows.append(dict(cohort="UIH", strategy=strategy, patient_id=str(pid), side=side,
                        task=task, label=int(label > 0) if task == "bin" else label,
                        prediction=0 if label == 1 else int(label > 0) if task == "bin" else label,
                        prob_positive=.4 if label == 1 else .8 if label > 0 else .1,
                        sex=None if pid == 0 else "Male", age=35, age_group="<40"))
    return pd.DataFrame(rows)


def test_clinical_truth_and_early_denominator():
    f = s.enrich_clinical(fixture())
    assert f[f.patient_id == "0"].disease_pattern.eq("Bilateral normal").all()
    assert f[f.patient_id == "1"].disease_pattern.eq("Unilateral ONFH").all()
    assert f[f.patient_id == "2"].stage_gap.eq(1).all()
    assert f[f.patient_id == "3"].stage_concordance.eq("Concordant").all()
    for task in ("bin", "stage"):
        g = f[(f.task == task) & (f.strategy == "zero_shot")]
        m = s.weighted_metrics(g, task, np.ones(len(g)))
        assert m["early_miss_rate" if task == "bin" else "early_to_normal"][0] == 1
        if task == "stage":
            assert m["early_recall"][0] == 0


def test_single_hip_and_conflicting_truth():
    f = fixture()
    f = f[~((f.patient_id == "1") & (f.side == "right"))]
    r = s.enrich_clinical(f)
    assert r[r.patient_id == "1"].disease_pattern.isna().all()
    f.loc[(f.patient_id == "0") & (f.task == "bin"), "label"] = 1
    with pytest.raises(ValueError, match="disagree"):
        s.enrich_clinical(f)


def test_pairing_audit(tmp_path):
    f = fixture()
    s.pairing_audit(f, tmp_path)
    f = f.drop(f.index[0])
    with pytest.raises(ValueError, match="pairing failed"):
        s.pairing_audit(f, tmp_path)
    assert len(pd.read_csv(tmp_path / "pairing_audit.csv")) == 2
    f = fixture()
    f.loc[0, "label"] = 1
    with pytest.raises(ValueError, match="pairing failed"):
        s.pairing_audit(f, tmp_path)


def test_cluster_weights_preserve_hips_and_reproducibility():
    f = s.enrich_clinical(fixture())
    f = f[(f.strategy == "zero_shot") & (f.task == "bin")]
    a = s.cluster_weights(f, 100, np.random.default_rng(42))
    assert np.array_equal(a[:, ::2], a[:, 1::2])
    assert np.array_equal(a, s.cluster_weights(f, 100, np.random.default_rng(42)))
    assert np.all(a.sum(axis=1) == len(f))


@pytest.mark.parametrize("task", ["bin", "stage"])
def test_weighted_bootstrap_matches_explicit_duplication(task):
    f = s.enrich_clinical(fixture())
    f = f[(f.strategy == "zero_shot") & (f.task == task)].reset_index(drop=True)
    w = np.array([2, 2, 1, 1, 3, 3, 1, 1])
    repeated = f.iloc[np.repeat(np.arange(len(f)), w)]
    result = s.weighted_metrics(f, task, w)
    if task == "bin":
        assert result["auroc"][0] == pytest.approx(roc_auc_score(repeated.label, repeated.prob_positive))
        assert result["auprc"][0] == pytest.approx(average_precision_score(repeated.label, repeated.prob_positive))
    else:
        assert result["qwk"][0] == pytest.approx(cohen_kappa_score(repeated.label, repeated.prediction, weights="quadratic", labels=[0, 1, 2]))
        assert result["macro_f1"][0] == pytest.approx(f1_score(repeated.label, repeated.prediction, average="macro", labels=[0, 1, 2]))


def test_single_class_empty_and_paired_identical():
    f = s.enrich_clinical(fixture())
    g = f[(f.task == "bin") & (f.patient_id == "0")]
    assert np.isnan(s.weighted_metrics(g, "bin", np.ones(len(g)))["auroc"][0])
    assert np.isnan(s.weighted_metrics(g.iloc[:0], "bin", np.zeros((2, 0)))["auroc"]).all()
    assert s.centered_bootstrap_p(np.zeros(100), 0) == 1
    assert s.centered_bootstrap_p(np.ones(100), 1) > 0


def test_external_mapping_and_cache(tmp_path):
    files = s.external_files(tmp_path)
    assert len(files) == 6
    assert all("all_predictions" not in str(p) for _, _, p in files)
    assert all("ensemble" in str(p) for _, strategy, p in files if strategy == "zero_shot")
    source = tmp_path / "input.csv"
    source.write_text("a")
    expected = s.fingerprint([source], {"threshold": .5})
    (tmp_path / "subgroup_predictions.csv").write_text("prediction")
    s.seal_cache(tmp_path, expected)
    assert s.cache_valid(tmp_path, expected)
    source.write_text("b")
    assert not s.cache_valid(tmp_path, s.fingerprint([source], {"threshold": .5}))
    assert not s.cache_valid(tmp_path, {"legacy": True})


def test_saved_split_uses_patient_id_not_source_order(tmp_path):
    api = SimpleNamespace(get_base_patient_id=lambda p: p, is_synthetic_sample=lambda p: p.startswith("gen"))
    test = dict(split="test", patient_id="b", source_index=999, sample_type="real",
                label_left_bin=1, label_right_bin=0, label_left_ELS=1, label_right_ELS=0)
    pd.DataFrame([test, dict(test, split="train", patient_id="a")]).to_csv(tmp_path / "fold_data_split.csv", index=False)
    samples = {"patient_id": ["b", "a"], "label_left_bin": [1, 0], "label_right_bin": [0, 0],
               "label_left_ELS": [1, 0], "label_right_ELS": [0, 0]}
    indices, _, ids = s.saved_indices(api, tmp_path / "model.pth", samples)
    assert indices.tolist() == [0] and ids == ["b"]
    samples["label_left_bin"][0] = 0
    with pytest.raises(ValueError, match="conflict"):
        s.saved_indices(api, tmp_path / "model.pth", samples)


def test_repeat_holdout_averages_only_observed_eligible_models():
    raw = pd.DataFrame([
        dict(cohort="internal", task="bin", patient_id="a", side="left", label=1,
             prediction=0, prob_positive=.2, fold=0, source_file="model0", split_source="split0"),
        dict(cohort="internal", task="bin", patient_id="a", side="left", label=1,
             prediction=1, prob_positive=.8, fold=2, source_file="model2", split_source="split2"),
        dict(cohort="internal", task="bin", patient_id="b", side="left", label=0,
             prediction=0, prob_positive=.1, fold=1, source_file="model1", split_source="split1")])
    result = s.aggregate_heldout(raw, "bin", .5).set_index("patient_id")
    assert result.loc["a", "prob_positive"] == .5
    assert result.loc["a", "heldout_model_count"] == 2
    assert result.loc["b", "heldout_model_count"] == 1
    assert result.loc["a", "fold"] == "0;2"
    with pytest.raises(ValueError, match="Duplicate"):
        s.aggregate_heldout(pd.concat([raw, raw.iloc[:1]]), "bin", .5)


def test_cross_task_conflict_audit_preserves_task_truth(tmp_path):
    raw = fixture()
    raw.loc[(raw.patient_id == "1") & (raw.side == "right") & (raw.task == "bin"), "label"] = 0
    frame = s.enrich_clinical(raw, tmp_path)
    conflict = frame[(frame.patient_id == "1") & (frame.side == "right")]
    assert conflict[conflict.task == "bin"].true_stage.isna().all()
    assert conflict[conflict.task == "stage"].true_stage.eq(1).all()
    assert frame[frame.patient_id == "1"].disease_pattern.isna().all()
    assert len(pd.read_csv(tmp_path / "cross_task_label_conflicts.csv")) == 1


def test_per_variable_missingness_and_empty_subgroup():
    frame = s.enrich_clinical(fixture())
    api = SimpleNamespace(BINARY_METRICS=("auroc",), STAGE_METRICS=("qwk",), holm_adjust=lambda p: np.asarray(p))
    metrics, _ = s.summarize(api, frame, 25, 2026)
    g = metrics[(metrics.strategy == "zero_shot") & (metrics.task == "bin") & (metrics.metric == "auroc")]
    assert g[g.subgroup == "Overall"].n_patients.iloc[0] == 4
    assert g[g.subgroup == "Male"].n_patients.iloc[0] == 3
    assert g[g.subgroup == "Bilateral normal"].n_patients.iloc[0] == 1
    assert g[g.subgroup == "Female"].estimate.isna().all()
    stage_empty = frame[(frame.sex == "Female") & (frame.task == "stage")]
    assert all(np.isnan(v).all() for v in s.weighted_metrics(stage_empty, "stage", np.zeros((2, 0))).values())


def test_identical_strategy_paired_bootstrap():
    frame = s.enrich_clinical(fixture())
    result = s.paired_differences(SimpleNamespace(holm_adjust=lambda p: np.asarray(p)), frame, 25, 2026)
    finite = result.dropna(subset=["difference"])
    assert np.allclose(finite.difference, 0)
    assert np.allclose(finite.ci_low.dropna(), 0)
    assert np.allclose(finite.p_value.dropna(), 1)


def test_bootstrap_moments_and_excel_export(tmp_path):
    mean, sd = s.bootstrap_moments([0.7, np.nan, 0.9])
    assert mean == pytest.approx(0.8)
    assert sd == pytest.approx(np.sqrt(0.02))

    internal = tmp_path / "internal"
    internal.mkdir()
    pd.DataFrame([{
        "cohort": "internal", "strategy": "siemens_internal_oof",
        "task": "bin", "subgroup_variable": "overall",
        "subgroup": "Overall", "metric": "auroc", "n_patients": 10,
        "n_hips": 20, "n_class_0": 8, "n_class_1": 12,
        "n_class_2": 0, "estimate": 0.81, "bootstrap_mean": 0.80,
        "bootstrap_sd": 0.05, "ci_low": 0.70, "ci_high": 0.90,
        "n_bootstrap_valid": 2000, "status": "estimable",
    }]).to_csv(internal / "subgroup_metrics.csv", index=False)
    output = s.export_excel_summary(tmp_path)
    assert output == tmp_path / "subgroup_metrics_mean_sd.xlsx"
    sheets = pd.read_excel(output, sheet_name=None)
    assert {"All metrics", "Definitions"} <= set(sheets)
    row = sheets["All metrics"].iloc[0]
    assert row["bootstrap_mean"] == pytest.approx(0.80)
    assert row["bootstrap_sd"] == pytest.approx(0.05)
    assert row["paper_mean_sd"] == "0.800 ± 0.050"
