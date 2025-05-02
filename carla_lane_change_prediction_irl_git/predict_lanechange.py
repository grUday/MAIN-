import pandas as pd
import numpy as np
import joblib

df = pd.read_csv("data/irl_features.csv")
X = df[["dx", "dy", "speed", "lane_diff"]].values
theta = joblib.load("models/irl_reward_weights.pkl")

# Compute reward for each state
rewards = X @ theta

# Predict action based on reward
df["reward"] = rewards
df["prediction"] = df["reward"].apply(lambda r: 1 if r > 0.2 else -1 if r < -0.2 else 0)

df.to_csv("data/predicted_irl_lane_changes.csv", index=False)
print("✅ IRL lane change predictions saved.")
