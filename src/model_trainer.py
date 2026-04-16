import torch
import torch.nn as nn
import torch.optim as optim
import clip
from src.emotion_model import EmotionModel

class VariancePenaltyLoss(nn.Module):
    def __init__(self, mse_weight=1.0, var_weight=0.5):
        super().__init__()
        self.mse = nn.MSELoss()
        self.mse_weight = mse_weight
        self.var_weight = var_weight

    def forward(self, predictions, targets):
        mse_loss = self.mse(predictions, targets)
        prediction_var = predictions.var(dim=0).mean()
        variance_penalty = torch.relu(0.05 - prediction_var)
        return self.mse_weight * mse_loss + self.var_weight * variance_penalty

class EmotionTrainer:
    """
    Trains EmotionModel (an MLP) on top of frozen CLIP image features.
    The DataLoader provides raw images (preprocessed for CLIP);
    we extract 512-dim embeddings here before passing to the MLP.
    """
    def __init__(self, device):
        self.device = device
        self.model = EmotionModel(device)
        self._clip_model, _ = clip.load("ViT-B/32", device=device)
        self._clip_model.eval()
        for p in self._clip_model.parameters():
            p.requires_grad = False
        self.criterion = VariancePenaltyLoss(mse_weight=1.0, var_weight=0.5)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=25, eta_min=1e-6)

    def _extract_features(self, images):
        with torch.no_grad():
            feats = self._clip_model.encode_image(images)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.float()

    def train(self, train_loader, epochs=25):
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for images, targets in train_loader:
                images = images.to(self.device)
                targets = targets.to(self.device)

                features = self._extract_features(images)
                predictions = self.model(features)
                loss = self.criterion(predictions, targets)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            self.scheduler.step()
            avg = total_loss / len(train_loader)
            print(f"Epoch {epoch + 1:>2}/{epochs} | Loss: {avg:.4f}")

    def save(self, path="student_model.pth"):
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")