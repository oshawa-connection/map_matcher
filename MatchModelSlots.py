import torch
from torch import nn as nn

from GridSearchParameterSet import GridSearchParameterSet


class MatchModelSlots(nn.Module):
    def __init__(self, parameterSet: GridSearchParameterSet):
        super(MatchModelSlots, self).__init__()
        self.cnn = parameterSet.cnn

        self.classifier = parameterSet.classifier

    def forward(self, img1, img2):
        feat1 = self.cnn(img1)
        feat2 = self.cnn(img2)
        diff = torch.abs(feat1 - feat2)
        out = self.classifier(diff.view(diff.size(0), -1))
        return out  # shape [B, 1]
