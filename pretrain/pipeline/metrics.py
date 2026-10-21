"""Task-native classification scores; no test-based threshold fitting."""

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score


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
    if classes == 2:
        output["auroc"] = float(roc_auc_score(labels, probabilities[:, 1]))
    else:
        output["weighted_f1"] = float(
            f1_score(labels, probabilities.argmax(-1), average="weighted")
        )
    return output
