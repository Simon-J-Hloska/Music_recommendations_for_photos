import torch
import torch.nn as nn
import clip


class EmotionEvaluator:
    """
    Evaluates the trained EmotionModel on held-out test images.
    Extracts CLIP features first, then runs the MLP.
    """
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self._clip_model, _ = clip.load("ViT-B/32", device=device)
        self._clip_model.eval()
        for p in self._clip_model.parameters():
            p.requires_grad = False
        self.mae = nn.L1Loss()
        self.mse = nn.MSELoss()

    def _extract_features(self, images):
        with torch.no_grad():
            feats = self._clip_model.encode_image(images)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.float()

    def evaluate(self, test_loader):
        self.model.eval()
        mae_total, mse_total = 0.0, 0.0
        n = len(test_loader)
        with torch.no_grad():
            for images, targets in test_loader:
                images = images.to(self.device)
                targets = targets.to(self.device)
                features = self._extract_features(images)
                preds = self.model(features)
                mae_total += self.mae(preds, targets).item()
                mse_total += self.mse(preds, targets).item()

        print("\n=== Evaluation Results ===")
        print(f"MAE  (lower is better): {mae_total / n:.4f}")
        print(f"RMSE (lower is better): {(mse_total / n) ** 0.5:.4f}")
        print("Scores are on a 0–1 scale (valence, energy, danceability)")