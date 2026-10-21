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
