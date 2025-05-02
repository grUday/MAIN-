import pandas as pd
import numpy as np

df = pd.read_csv("data/irl_expert_traj.csv")
vehicle_ids = df["id"].unique()
features = []

for vid in vehicle_ids:
    track = df[df["id"] == vid].sort_values("frame").reset_index(drop=True)
    for i in range(1, len(track)):
        prev = track.iloc[i - 1]
        curr = track.iloc[i]
        action = 1 if curr["laneId"] > prev["laneId"] else -1 if curr["laneId"] < prev["laneId"] else 0
        feat = [
            curr["x"] - prev["x"],  # dx
            curr["y"] - prev["y"],  # dy
            np.linalg.norm([curr["vx"], curr["vy"]]),  # speed
            curr["laneId"] - prev["laneId"],  # lane delta
            action
        ]
        features.append(feat)

df_out = pd.DataFrame(features, columns=["dx", "dy", "speed", "lane_diff", "action"])
df_out.to_csv("data/irl_features.csv", index=False)
print("✅ IRL features saved to data/irl_features.csv")
