import torch

from pretrain.pipeline.metrics import LossTotals


def test_batched_loss_reads_preserve_python_precision_order_and_detach():
    rows = [
        torch.tensor(values, requires_grad=True)
        for values in (
            [1e-9, 4.25, 1e8],
            [1.2, 0.3, 2.1],
            [3.7, 8.3, 5.5],
        )
    ]
    weights = [4, 4, 1]
    expected = [0.0, 0.0, 0.0]
    totals = LossTotals()
    for weight, row in zip(weights, rows):
        totals.add(weight, *row.unbind())
        for index, value in enumerate(row):
            expected[index] += float(value.detach()) * weight
    assert totals.sums() == tuple(expected)
    assert all(row.grad_fn is None and not row.requires_grad for row in totals.rows)
    assert all(row.grad is None for row in rows)


def test_binary_metrics_keep_accuracy_and_f1_averages_distinct():
    import numpy as np
    import pytest
    from pretrain.pipeline.metrics import classification

    labels = np.array([0, 0, 0, 0, 0, 1])
    positive = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    scores = classification(labels, np.column_stack([1 - positive, positive]))
    assert scores["acc"] == pytest.approx(2 / 3)
    assert scores["balanced_accuracy"] == pytest.approx(0.8)
    assert scores["f1_score_binary"] == pytest.approx(0.5)
    assert scores["f1_score_micro"] == pytest.approx(2 / 3)
    assert scores["f1_score_macro"] == pytest.approx(0.625)
    assert scores["weighted_f1"] == pytest.approx(17 / 24)
    assert scores["precision_macro"] == pytest.approx(2 / 3)
    assert scores["recall_macro"] == pytest.approx(0.8)


def test_multiclass_f1_has_explicit_averages_without_binary_f1():
    import numpy as np
    from sklearn.metrics import f1_score
    from pretrain.pipeline.metrics import classification

    labels = np.array([0, 0, 0, 1, 1, 2])
    predicted = np.array([0, 0, 1, 1, 2, 2])
    probabilities = np.eye(3)[predicted] * 0.8 + 0.2 / 3
    scores = classification(labels, probabilities)
    assert "f1_score_binary" not in scores
    for average, key in [
        ("micro", "f1_score_micro"),
        ("macro", "f1_score_macro"),
        ("weighted", "weighted_f1"),
    ]:
        assert scores[key] == f1_score(labels, predicted, average=average)
