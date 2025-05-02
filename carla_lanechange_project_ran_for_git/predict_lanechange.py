import pandas as pd
import joblib

model = joblib.load("models/lane_change_model.pkl")
df = pd.read_csv("data/processed_features.csv")

X = df[["dx", "dy", "speed"]]
preds = model.predict(X)

df["prediction"] = preds
df.to_csv("data/predicted_lane_changes.csv", index=False)
print("✅ Predictions saved: data/predicted_lane_changes.csv")
