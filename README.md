# 🎵 PhotoSync — Music Matched to Your Moments

Drop a photo. Get music that fits.

---

## What it does

PhotoSync analyzes the mood of any photo and returns Spotify tracks that emotionally match it — using the same audio dimensions Spotify uses internally: **valence**, **energy**, and **danceability**.

```
your photo  →  mood analysis  →  [valence, energy, dance]  →  matching tracks
```

---

## How to run it

**Requirements:** Python 3.10+, CUDA optional (runs on CPU)

```bash
pip install -r requirements.txt
python src/main.py
```

A window opens. **Drag and drop** one or more photos into it. Results appear within seconds.

---

## What you get back

For each photo, the top 5 matched tracks with:

| Field | Example |
|---|---|
| Track name | *Blinding Lights* |
| Artist | The Weeknd |
| Genre | pop |
| Valence | 0.73 |
| Energy | 0.80 |
| Danceability | 0.51 |

---

## The mood space

Every photo and every song gets placed in the same 3D space:

```
         high energy
              │
  dark ───────┼─────── happy       (valence axis)
              │
         low energy

+ danceability as the third dimension
```

A sunny crowded beach → high valence, high energy, high dance → upbeat pop  
A misty empty trail → low valence, low energy, low dance → ambient or slow folk

---

## Example matches

| Photo type | Valence | Energy | Dance | Matched sound |
|---|---|---|---|---|
| Crowded festival | 0.84 | 0.91 | 0.88 | Electronic / Dance |
| Rainy window | 0.21 | 0.28 | 0.23 | Lo-fi / Ambient |
| Golden hour portrait | 0.76 | 0.52 | 0.55 | Indie / Soul |
| Night city street | 0.31 | 0.68 | 0.47 | Hip-hop / Dark pop |
| Mountain sunrise | 0.58 | 0.22 | 0.20 | Post-rock / Classical |

---

## Training your own model

If you want to retrain on your own photo set:

```bash
# 1. Put raw photos in data/images/training/raw/
# 2. Put your Spotify dataset CSV in data/music/dataset.csv
# 3. Run the training pipeline
python src/main.py --train   # calls main1() internally
```

The model trains in ~30 epochs. Expect a few minutes on CPU, under a minute on GPU.

The trained model is saved as `student_model.pth` and loaded automatically on next run.

---

## Dataset format

Your `dataset.csv` must contain at minimum:

```
track_id, track_name, artists, album_name, track_genre, valence, energy, danceability, popularity
```

Compatible with the [Spotify Tracks Dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset) on Kaggle (114k tracks).

---

## Project structure

```
├── src/
│   ├── main.py                   entry point
│   ├── emotion_model.py          the neural network
│   ├── dataset_image_labeler.py  generates training labels via CLIP
│   ├── model_trainer.py          training loop
│   ├── emotion_evaluator.py      test set evaluation
│   ├── dataset_matcher.py        finds matching songs
│   ├── image_dataset_builder.py  data loading
│   ├── Image_normalization.py    photo preprocessing
│   └── ui.py                     drag-and-drop interface
├── data/
│   ├── images/training/          training photos
│   ├── images/user_images/       temporary drop folder (auto-cleared)
│   └── music/dataset.csv         your Spotify track library
└── student_model.pth             trained model weights
```

---

## Under the hood (one paragraph)

PhotoSync uses **CLIP** (OpenAI's vision model) to understand what a photo is about, then a small neural network maps that understanding into Spotify's `[valence, energy, danceability]` space. Training labels are generated automatically — CLIP compares each photo against 32 mood descriptions (*"joyful beach party"*, *"melancholic rainy window"*, etc.) and assigns emotional scores grounded in real Spotify cluster data. No manual labeling needed.

---

*Built with PyTorch · CLIP ViT-B/32 · scikit-learn · tkinterdnd2*
