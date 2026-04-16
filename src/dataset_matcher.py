"""
dataset_matcher.py

Given a predicted [valence, energy, danceability] vector from the model,
finds the closest matching tracks from the local dataset.csv file.
"""
import numpy as np
import pandas as pd
from collections import deque

class DatasetMatcher:
    def __init__(self, csv_path="data/music/dataset.csv"):
        self.csv_path = csv_path
        self._df = None
        self._vectors = None
        self.popularity_norm = None
        self._recent_tracks = deque(maxlen=40)
        self._name_to_indices = None

    def load(self):
        """Load the dataset and build the search index."""
        print(f"Loading track index from {self.csv_path}...")
        df = pd.read_csv(self.csv_path)
        df = df.dropna(subset=["valence", "energy", "danceability", "track_name", "artists"])
        df = df.drop_duplicates(subset="track_id")
        self._df = df.reset_index(drop=True)
        self._vectors = self._df[["valence", "energy", "danceability"]].values.astype(np.float32)
        popularity = self._df["popularity"].fillna(0).to_numpy(dtype=np.float32)
        self.popularity_norm = popularity / 100.0
        print(f"  Index ready: {len(self._df)} unique tracks")

    def match(self, predicted_vector, top_k=5):
        """
        Find the top_k closest tracks to a predicted [valence, energy, dance] vector.

        predicted_vector: list or np.array of shape [3]
        Returns: list of dicts sorted by distance (closest first)
        """
        if self._vectors is None:
            raise RuntimeError("Call load() before match()")
        query = np.array(predicted_vector, dtype=np.float32)

        # Weighted Euclidean
        repetition_penalty = 10.0
        popularity_weight = 0.13
        weights = np.array([1.5, 1.0, 1.0], dtype=np.float32)
        diff = (self._vectors - query) * weights
        distances = np.sqrt((diff ** 2).sum(axis=1))
        scores = distances - (popularity_weight * self.popularity_norm)
        self._name_to_indices = {}
        for i, name in enumerate(self._df["track_name"]):
            self._name_to_indices.setdefault(name, []).append(i)
        if self._recent_tracks:
            recent_list = list(self._recent_tracks)
            from collections import Counter
            counts = Counter(recent_list)
            for track_name, count in counts.items():
                indices = self._name_to_indices.get(track_name, [])
                if indices:
                    scores[indices] *= (repetition_penalty ** count)

        top_idx = np.argsort(scores)[:top_k]
        results = []
        for idx in top_idx:
            row = self._df.iloc[idx]
            self._recent_tracks.append(row["track_name"])
            results.append({
                "track_name": row["track_name"],
                "artists": row["artists"],
                "album_name": row.get("album_name", ""),
                "genre": row.get("track_genre", ""),
                "valence": float(row["valence"]),
                "energy": float(row["energy"]),
                "danceability": float(row["danceability"]),
                "popularity": int(row.get("popularity", 0)),
                "distance": float(distances[idx]),
                "score": float(scores[idx]),
            })
        return results

    def match_from_image(self, model, clip_model, preprocess, image_path, device, top_k=5):
        """
        End-to-end: image path → top_k matched tracks.
        """
        import torch
        from PIL import Image
        img = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            feats = clip_model.encode_image(img)
            feats /= feats.norm(dim=-1, keepdim=True)
            pred = model(feats.float()).cpu().numpy()[0]

        print(f"Predicted → valence: {pred[0]:.3f} | energy: {pred[1]:.3f} | dance: {pred[2]:.3f}")
        return self.match(pred, top_k=top_k)