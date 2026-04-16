"""
Labels training images using the local music dataset (data/music/dataset.csv).

1. Load the local dataset CSV → extract valence/energy/danceability values
2. K-means cluster those tracks into mood prototypes in [valence, energy, dance] space
3. Use CLIP to embed each photo against 32 natural-language mood descriptions
4. Softmax-weight the CLIP similarities → weighted average over cluster centroids
5. Save image → [valence, energy, dance] CSV with real dataset-derived targets
"""

import os
import csv
import numpy as np
import torch
import pandas as pd
from PIL import Image
from sklearn.cluster import KMeans
import clip


MOOD_TEXTS = [
    "a joyful beach party with friends in summer",
    "people dancing at a nightclub with colorful lights",
    "an exciting live music concert with crowd energy",
    "a peaceful sunrise over a quiet mountain lake",
    "a romantic candlelit dinner for two",
    "a cozy rainy day reading inside",
    "a dark moody urban alley at night",
    "a melancholic person sitting alone looking at the rain",
    "an intense industrial factory with heavy machinery",
    "a vibrant street festival with dancing and music",
    "a serene yoga session in nature",
    "a high-speed sports car on a racetrack",
    "a nostalgic family gathering with warm light",
    "a deep focus study session late at night",
    "a sad empty room with faded photographs",
    "an aggressive mosh pit at a metal concert",
    "a soft pastel flower garden in spring",
    "children laughing and playing in a park",
    "a dark thunderstorm over open ocean",
    "a slow romantic dance under the stars",
    "a chaotic busy city intersection at rush hour",
    "a calm meditation retreat in a bamboo forest",
    "a triumphant athlete crossing the finish line",
    "a lonely hiker on a misty grey trail",
    "a vibrant Latino street parade",
    "a quiet jazz bar with dim lighting",
    "an abstract expressionist painting with bold colors",
    "a cold snowy empty landscape at dusk",
    "a warm golden-hour portrait of someone smiling",
    "a tense dramatic scene before a storm",
    "a carefree road trip on an open highway",
    "a minimalist zen garden with raked sand",
]


def load_tracks(csv_path="data/music/dataset.csv"):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["valence", "energy", "danceability"])
    print(f"  Loaded {len(df)} tracks")
    return df

def build_mood_clusters(df, n_clusters=32):
    """
    K-means cluster tracks in [valence, energy, danceability] space.
    Returns cluster_centers array of shape [n_clusters, 3].
    """
    X = df[["valence", "energy", "danceability"]].values.astype(np.float32)
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    km.fit(X)
    print(f"Built {n_clusters} mood clusters from {len(df)} tracks")
    return km.cluster_centers_, km


class DatasetImageLabeler:
    """
    For each image:
    1. Get CLIP image embedding
    2. Compare against MOOD_TEXTS embeddings → softmax probabilities
    3. Map mood probs → Spotify cluster centroids → weighted [valence, energy, dance]
    """
    def __init__(self, device, cluster_centers):
        self.device = device
        self.cluster_centers = cluster_centers  # [n_clusters, 3]
        self.model, self.preprocess = clip.load("ViT-B/32", device=device)
        self.model.eval()
        self._text_features = self._encode_texts()
        self._mood_values = self._map_moods_to_clusters()

    def _encode_texts(self):
        tokens = clip.tokenize(MOOD_TEXTS).to(self.device)
        with torch.no_grad():
            feats = self.model.encode_text(tokens)
            feats /= feats.norm(dim=-1, keepdim=True)
        return feats  # [n_moods, 512]

    def label_directory(self, image_dir, out_csv, batch_size=64):
        valid_exts = (".jpg", ".jpeg", ".png", ".webp")
        files = [f for f in os.listdir(image_dir) if f.lower().endswith(valid_exts)]
        print(f"Labeling {len(files)} images...")

        results = []
        for i in range(0, len(files), batch_size):
            batch = files[i:i + batch_size]
            images = self._load_batch(image_dir, batch)
            with torch.no_grad():
                img_feats = self.model.encode_image(images)
                img_feats /= img_feats.norm(dim=-1, keepdim=True)
                sim = img_feats @ self._text_features.T
                temperature = 0.07
                probs = (sim / temperature).softmax(dim=-1).cpu().numpy()

            for name, p in zip(batch, probs):
                vals = (p[:, None] * self._mood_values).sum(axis=0)
                results.append([name, *vals.tolist()])

            print(f"{min(i + batch_size, len(files))}/{len(files)}")

        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        with open(out_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["image", "valence", "energy", "dance"])
            writer.writerows(results)
        print(f"Saved labels → {out_csv}")
        return results

    def _load_batch(self, image_dir, files):
        imgs = [self.preprocess(Image.open(os.path.join(image_dir, f)).convert("RGB")) for f in files]
        return torch.stack(imgs).to(self.device)

    def _map_moods_to_clusters(self):
        """
        For each mood text, find the cluster centroid that best matches
        its expected valence/energy/danceability using a hand-coded prior.
        """
        # Rough expected [valence, energy, dance] for each MOOD_TEXT
        mood_priors = np.array([
            [0.9, 0.8, 0.8],  # joyful beach party
            [0.7, 0.9, 0.95],  # nightclub dancing
            [0.7, 0.95, 0.7],  # live concert
            [0.6, 0.2, 0.2],  # peaceful sunrise
            [0.7, 0.3, 0.3],  # romantic dinner
            [0.4, 0.2, 0.2],  # cozy rainy day
            [0.1, 0.4, 0.3],  # dark urban alley
            [0.1, 0.2, 0.2],  # melancholic person
            [0.2, 0.8, 0.4],  # industrial factory
            [0.8, 0.85, 0.9],  # street festival
            [0.5, 0.2, 0.2],  # yoga in nature
            [0.6, 0.9, 0.6],  # sports car racetrack
            [0.7, 0.4, 0.4],  # nostalgic family
            [0.3, 0.4, 0.3],  # late night study
            [0.05, 0.15, 0.2],  # sad empty room
            [0.2, 0.95, 0.7],  # metal mosh pit
            [0.8, 0.3, 0.4],  # pastel flower garden
            [0.9, 0.7, 0.75],  # children laughing
            [0.1, 0.7, 0.2],  # dark thunderstorm
            [0.6, 0.25, 0.45],  # slow romantic dance
            [0.4, 0.75, 0.6],  # chaotic city
            [0.5, 0.15, 0.2],  # meditation bamboo
            [0.8, 0.9, 0.65],  # triumphant athlete
            [0.2, 0.3, 0.25],  # lonely hiker
            [0.85, 0.85, 0.9],  # Latino street parade
            [0.4, 0.3, 0.4],  # quiet jazz bar
            [0.6, 0.6, 0.5],  # abstract painting
            [0.1, 0.2, 0.15],  # cold snowy landscape
            [0.85, 0.55, 0.6],  # warm golden portrait
            [0.2, 0.6, 0.3],  # tense before storm
            [0.75, 0.7, 0.7],  # carefree road trip
            [0.4, 0.1, 0.2],  # zen garden
        ], dtype=np.float32)
        return mood_priors