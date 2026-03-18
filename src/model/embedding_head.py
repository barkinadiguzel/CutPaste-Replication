import torch.nn as nn
import torch.nn.functional as F

class EmbeddingHead(nn.Module):
    def __init__(self, in_features=512, out_features=128):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x):
        x = self.fc(x)
        x = F.normalize(x, dim=1)  
        return x
