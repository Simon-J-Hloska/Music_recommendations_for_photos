import torch.nn as nn


class EmotionModel(nn.Module):
    """
    Lightweight MLP that takes a 512-dim CLIP image embedding
    and regresses to [valence, energy, danceability] in [0, 1].
    """
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.net = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 3),
            nn.Sigmoid(),
        )
        self.to(device)

    def forward(self, x):
        return self.net(x)