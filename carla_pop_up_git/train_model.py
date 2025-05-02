import pandas as pd
import numpy as np
import joblib

df = pd.read_csv("data/irl_features.csv")
X = df[["dx", "dy", "speed", "lane_diff"]].values
A = df["action"].values

# Feature expectations from expert
mu_expert = X.mean(axis=0)

# Initialize reward weights randomly
theta = np.random.rand(X.shape[1])
alpha = 0.1

def softmax_policy(X, theta):
    rewards = X @ theta
    probs = np.exp(rewards) / np.sum(np.exp(rewards))
    return probs

# Train reward weights
for epoch in range(100):
    probs = softmax_policy(X, theta)
    mu_model = np.average(X, axis=0, weights=probs)
    grad = mu_expert - mu_model
    theta += alpha * grad
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: |Grad| = {np.linalg.norm(grad):.4f}")

joblib.dump(theta, "models/irl_reward_weights.pkl")
print("✅ Reward function trained and saved.")
