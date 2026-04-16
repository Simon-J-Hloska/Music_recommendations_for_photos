import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from src.img_emotion_dataset import ImageEmotionDataset
from sklearn.preprocessing import MinMaxScaler

class EmotionDatasetBuilder:
    def __init__(self, csv_path, image_dir, transform, batch_size=64):
        self.csv_path = csv_path
        self.image_dir = image_dir
        self.transform = transform
        self.batch_size = batch_size

    def build(self):
        df = pd.read_csv(self.csv_path)
        df[["valence", "energy", "dance"]] = MinMaxScaler().fit_transform(df[["valence", "energy", "dance"]])
        train_df, test_df = train_test_split(df, test_size=0.4, random_state=50)

        train_dataset = ImageEmotionDataset(train_df, self.image_dir, self.transform)
        test_dataset = ImageEmotionDataset(test_df, self.image_dir, self.transform)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size)

        print(f"Dataset: {len(train_df)} train, {len(test_df)} test samples")
        return train_loader, test_loader