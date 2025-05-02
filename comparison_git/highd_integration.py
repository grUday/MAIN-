import pandas as pd
import os

class HighDData:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        print("📦 Loading HighD dataset...")
        self.tracks = self.load_csv("01_tracks.csv")
        self.tracks_meta = self.load_csv("01_tracksMeta.csv")
        self.recording_meta = self.load_csv("01_recordingMeta.csv")
        print("✅ HighD dataset loaded successfully.")

    def load_csv(self, filename):
        file_path = os.path.join(self.dataset_path, filename)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"🚨 File not found: {file_path}")
        print(f"✅ Loading {file_path}...")
        return pd.read_csv(file_path)

    def get_sample_data(self, count=5, offset=0):
        df = self.tracks.copy()
        df = df[df['frame'] == df['frame'].min()]
        df = df.iloc[offset:offset+count]
        return df[['id', 'x', 'y', 'laneId']]

    def save_to_pickle(self, df, filename):
        import pickle
        with open(filename, "wb") as f:
            pickle.dump(df, f)
