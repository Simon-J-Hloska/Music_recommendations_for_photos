"""
main.py — Photo-Music pipeline

Full flow:
  1. Resize/normalize raw photos
  2. Fetch real Spotify tracks → cluster into mood prototypes
  3. Label each photo using CLIP + Spotify cluster centroids (real valence/energy/dance targets)
  4. Train EmotionModel MLP on those real labels
  5. Evaluate on held-out test set
  6. (Optional) Match a new photo to Spotify tracks
"""
import os
import sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from src.Image_normalization import ImageNormalizer
from src.dataset_image_labeler import load_tracks, build_mood_clusters, DatasetImageLabeler
from src.emotion_model import EmotionModel
from src.image_dataset_builder import EmotionDatasetBuilder
from src.model_trainer import EmotionTrainer
from src.emotion_evaluator import EmotionEvaluator
from src.dataset_matcher import DatasetMatcher
from src.ui import ImageDrop

RAW_DIR      = "../data/images/training/raw"
RESIZED_DIR  = "../data/images/training/resized"
RESULT_CSV   = "../data/images/training/images_w_emotion.csv"

raw_directory = "../data/images/user_images/raw"
resized_directory = "../data/images/user_images/resized"
MUSIC_CSV    = "../data/music/dataset.csv"
MODEL_PATH   = "student_model.pth"
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    drop = ImageDrop()
    tracks_df, cluster_centers, labeler, preprocess, clip_model, student_model, matcher = prepare()
    while True:
        drop.run()
        #calculating?

        normalizer = ImageNormalizer(raw_directory, resized_directory)
        normalizer.run()

        matches_dictionary = compute_images(preprocess, clip_model, student_model, resized_directory, labeler, matcher)
        for image_name in matches_dictionary.keys():
            matches = matches_dictionary[image_name]
            drop.show_results(matches, image_name)
        action = drop.show_action_buttons()
        clean_folders([raw_directory, resized_directory])
        match action:
            case "close":
                break
            case "restart":
                drop.reset()
                continue

def clean_folders(folder_path_list):
    for folder in folder_path_list:
        folder = Path(folder)
        for item in folder.iterdir():
            if item.is_file():
                item.unlink()

def prepare():
    tracks_df = load_tracks(MUSIC_CSV)
    cluster_centers, _ = build_mood_clusters(tracks_df, n_clusters=32)
    labeler = DatasetImageLabeler(device=DEVICE, cluster_centers=cluster_centers)
    clip_model = labeler.model
    preprocess = labeler.preprocess
    student_model = EmotionModel(device=DEVICE)
    student_model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    student_model.eval()
    matcher = DatasetMatcher(csv_path=MUSIC_CSV)
    matcher.load()
    return tracks_df, cluster_centers, labeler, preprocess, clip_model, student_model, matcher

def compute_images(preprocess, clip_model, student_model, dir_path, labeler, matcher):
    matches_dictionary = {}
    clip_prediction = None
    for image_name in os.listdir(dir_path):
        image_path = os.path.join(dir_path, image_name)
        img = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            img_feats = clip_model.encode_image(img)
            img_feats /= img_feats.norm(dim=-1, keepdim=True)

        with torch.no_grad():
            student_prediction = (student_model(img_feats.float()).cpu().numpy()[0])
        #clip_prediction = clip_predict(img_feats, labeler) #function to compute clip prediction
        calculate_error(clip_prediction, student_prediction, image_name)

        matches = matcher.match(student_prediction, top_k=5)
        matches_dictionary[image_name] = matches
    return matches_dictionary

def calculate_error(clip_prediction, student_prediction, img_name):
    if clip_prediction is None:
        return
    #distance between them in space <0;1.732>
    L2_distance = np.linalg.norm(student_prediction - clip_prediction)
    #direction similarity (angle) <0;1>
    cos_sim = (np.dot(student_prediction, clip_prediction) /
               (np.linalg.norm(student_prediction) * np.linalg.norm(clip_prediction)))
    print(f"image: {img_name}:\n")
    print(f"Cosine similarity: {cos_sim}")
    print(f"L2 distance: {L2_distance}")

def clip_predict(img_feats, labeler):
    with torch.no_grad():
        sim = img_feats @ labeler._text_features.T
        probs = sim.softmax(dim=-1).cpu().numpy()[0]
    return (probs[:, None] * labeler._mood_values).sum(axis=0)



def main1():
    print("\n=== Step 1: Normalizing images ===")
    normalizer = ImageNormalizer(input_dir=RAW_DIR, output_dir=RESIZED_DIR)
    normalizer.run()

    print("\n=== Step 2: Fetching Spotify mood data ===")
    tracks_df = load_tracks(MUSIC_CSV)
    cluster_centers, km = build_mood_clusters(tracks_df, n_clusters=32)

    print("\n=== Step 3: Labeling images with CLIP ===")
    labeler = DatasetImageLabeler(device=DEVICE, cluster_centers=cluster_centers)
    labeler.label_directory(RESIZED_DIR, RESULT_CSV)

    print("\n=== Step 4: Training ===")
    builder = EmotionDatasetBuilder(RESULT_CSV, RESIZED_DIR, transform=labeler.preprocess)
    train_loader, test_loader = builder.build()
    trainer = EmotionTrainer(device=DEVICE)
    trainer.train(train_loader, epochs=30)
    trainer.save(MODEL_PATH)

    print("\n=== Step 5: Evaluation ===")
    evaluator = EmotionEvaluator(trainer.model, device=DEVICE)
    evaluator.evaluate(test_loader)

    print("\n=== Step 6: Building Spotify match index ===")
    matcher = DatasetMatcher(csv_path=MUSIC_CSV)
    matcher.load()
    sys.exit(0)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(e)