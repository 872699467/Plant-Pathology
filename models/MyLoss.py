import torch
import torch.nn as nn


class DenseCrossEntropy(nn.Module):

    def __init__(self):
        super(DenseCrossEntropy, self).__init__()

    def forward(self, y_pred, y_true):
        y_pred = y_pred.float()
        y_true = y_true.float()

        logprobs = torch.log_softmax(y_pred, dim=-1)

        loss = -y_true * logprobs
        loss = torch.sum(loss)

        return loss.mean()
