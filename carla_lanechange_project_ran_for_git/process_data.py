import pandas as pd
import numpy as np

df = pd.read_csv("data/carla_traj.csv")
vehicle_ids = df["id"].unique()
all_features = []

for vid in vehicle_ids:
    track = df[df["id"] == vid].sort_values("frame").reset_index(drop=True)
    for i in range(1, len(track)):
        prev = track.iloc[i - 1]
        curr = track.iloc[i]
        label = 1 if curr["laneId"] > prev["laneId"] else -1 if curr["laneId"] < prev["laneId"] else 0
        feature = [
            curr["x"] - prev["x"],
            curr["y"] - prev["y"],
            np.linalg.norm([curr["vx"], curr["vy"]]),
            label
        ]
        all_features.append(feature)

df_out = pd.DataFrame(all_features, columns=["dx", "dy", "speed", "lane_change_label"])
df_out.to_csv("data/processed_features.csv", index=False)
print("✅ Features saved: data/processed_features.csv")
