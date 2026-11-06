"""Task-native classification scores; no test-based threshold fitting."""

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


class LossTotals:
    """Read detached microbatch losses once per logical update.

    Python sums retain the original reduction order and double precision.
    Only the scalar transfer is batched; the training objective is unchanged.
    """

    def __init__(self):
        self.weights = []
        self.rows = []

    def add(self, samples, *losses):
        self.weights.append(samples)
        self.rows.append(torch.stack([value.detach() for value in losses]))

    def sums(self):
        values = torch.stack(self.rows).cpu().tolist()
        return tuple(
            sum(weight * row[index] for weight, row in zip(self.weights, values))
            for index in range(len(values[0]))
        )


def classification(labels, probabilities):
    labels, probabilities = np.asarray(labels), np.asarray(probabilities)
    if labels.ndim != 1 or probabilities.ndim != 2 or len(labels) != len(probabilities):
        raise ValueError("Label and probability shapes differ")
    classes = probabilities.shape[1]
    if not np.array_equal(np.unique(labels), np.arange(classes)):
        raise ValueError("Evaluation must contain every declared class")
    if not np.isfinite(probabilities).all() or (probabilities < 0).any():
        raise ValueError("Invalid classification probabilities")
    if not np.allclose(probabilities.sum(-1), 1, atol=1e-4):
        raise ValueError("Probabilities do not sum to one")
    output = {"balanced_accuracy": float(balanced_accuracy_score(labels, probabilities.argmax(-1)))}
    predictions = probabilities.argmax(-1)
    output.update(
        loss=float(log_loss(labels, probabilities)),
        acc=float(accuracy_score(labels, predictions)),
        f1_score_micro=float(f1_score(labels, predictions, average="micro", zero_division=0)),
        f1_score_macro=float(f1_score(labels, predictions, average="macro", zero_division=0)),
        weighted_f1=float(f1_score(labels, predictions, average="weighted", zero_division=0)),
        precision_macro=float(
            precision_score(labels, predictions, average="macro", zero_division=0)
        ),
        recall_macro=float(recall_score(labels, predictions, average="macro", zero_division=0)),
    )
    if classes == 2:
        output["f1_score_binary"] = float(
            f1_score(labels, predictions, average="binary", pos_label=1, zero_division=0)
        )
        output["auroc"] = float(roc_auc_score(labels, probabilities[:, 1]))
        output["auprc"] = float(average_precision_score(labels, probabilities[:, 1]))
    else:
        output["auroc"] = float(
            roc_auc_score(labels, probabilities, multi_class="ovr", average="macro")
        )
        output["auprc"] = float(
            average_precision_score(np.eye(classes)[labels], probabilities, average="macro")
        )
    return output
