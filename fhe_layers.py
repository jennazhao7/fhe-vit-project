import torch
import torch.nn as nn

class FHELinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.inner = nn.Linear(in_features, out_features, bias)

    def forward(self, x):
        # This is where OpenFHE logic will eventually go
        print("FHELinear: input shape =", x.shape)
        return self.inner(x)