# predictive_models/keyboard/train_compare.py

from sklearn.metrics import (
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
)

import torch
import torch.nn.functional as F


MODELS = {
    "gru": "v1_gru.KeyboardGRU",
    "tcn": "v2_tcn.KeyboardTCN",
    "transformer": "v3_transformer.KeyboardTransformer",
    "hybrid": "v4_hybrid.KeyboardHybrid",
}


class JointLoss(torch.nn.Module):

    def forward(self, output, tlx_targets, state_targets):

        factor_loss = F.huber_loss(
            output[:, :5],
            tlx_targets,
        )

        state_loss = F.cross_entropy(
            output[:, 5:10],
            state_targets,
        )

        total = 0.4 * factor_loss + 0.6 * state_loss

        return total


def evaluate(y_true, y_pred):

    return {
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }