import numpy as np

from scripts.train_pb_full_spectrum import (
    classification_metrics,
    create_run_directory,
)


def test_classification_metrics_use_explicit_confusion_matrix_accuracy() -> None:
    metrics = classification_metrics(
        np.asarray([1, 1, 0, 0]),
        np.asarray([0.8, 0.2, 0.7, 0.1]),
    )

    assert metrics["confusion_matrix"] == {"tn": 1, "fp": 1, "fn": 1, "tp": 1}
    assert metrics["accuracy"] == (metrics["tp"] + metrics["tn"]) / (
        metrics["tp"] + metrics["tn"] + metrics["fp"] + metrics["fn"]
    )
    assert metrics["precision"] == 0.5
    assert metrics["recall_sensitivity"] == 0.5
    assert metrics["specificity"] == 0.5
    assert metrics["f1"] == 0.5
    assert metrics["roc_auc"] == 0.75


def test_roc_auc_is_unavailable_for_single_class_validation_labels() -> None:
    metrics = classification_metrics(np.asarray([1, 1]), np.asarray([0.8, 0.3]))

    assert metrics["roc_auc"] is None


def test_each_run_gets_a_new_output_directory(tmp_path) -> None:
    base_dir = tmp_path / "pb_presence_absence"

    first = create_run_directory(base_dir)
    second = create_run_directory(base_dir)

    assert first.is_dir()
    assert second.is_dir()
    assert first != second