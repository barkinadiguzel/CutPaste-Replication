import torch.nn as nn

class CutPasteClassifier(nn.Module):
    def __init__(self, in_features=512):
        super().__init__()
        self.fc = nn.Linear(in_features, 2) 

    def forward(self, x):
        return self.fc(x)
