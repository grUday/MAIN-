import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate

# === OUTPUT DIR ===
os.makedirs("output", exist_ok=True)

# === Simulated X-axis (Distance) ===
x = np.linspace(0, 150, 50)

# === Smooth curve function ===
def smooth_curve(start_y, end_y, steps=50):
    return start_y + (end_y - start_y) * np.sin(np.linspace(0, np.pi, steps)) / 2

# === Trajectories for each model ===
y_idm = np.concatenate([
    np.zeros(15),                         # Keep
    smooth_curve(0, 0.9, 20),             # Gentle left
    np.ones(15) * 0.9                     # Hold
])

y_krauss = np.concatenate([
    np.zeros(10),                         # Keep
    smooth_curve(0, 1.2, 20),             # Slightly more aggressive left
    np.ones(20) * 1.2                     # Hold
])

y_wiedemann = np.concatenate([
    np.zeros(30),                         # Keep
    smooth_curve(0, -0.6, 10),            # Right lane change
    np.ones(10) * -0.6                    # Hold
])

y_rl = np.concatenate([
    smooth_curve(0, -1.0, 10),            # Initial right
    smooth_curve(-1.0, 1.0, 20),          # Left lane change
    smooth_curve(1.0, 0.0, 20)            # Return to center
])

# Align all with x
x_vals = np.linspace(0, 150, len(y_idm))

# === Plotting Lane Trajectories ===
plt.figure(figsize=(12, 6))
plt.plot(x_vals, y_idm, label='IDM')
plt.plot(x_vals, y_krauss, label='Krauss')
plt.plot(x_vals, y_wiedemann, label='Wiedemann')
plt.plot(x_vals, y_rl, label='RL-Aggressive')

plt.axhline(0.0, linestyle='--', color='gray', linewidth=0.5)
plt.axhline(0.5, linestyle='--', color='gray', linewidth=0.5)
plt.axhline(-0.5, linestyle='--', color='gray', linewidth=0.5)

plt.title("Lane Trajectories for Each Model (Realistic)")
plt.xlabel("Distance (X)")
plt.ylabel("Lateral Position (Y)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("output/lane_trajectories.png")
plt.show()

# === Simulate Predictions ===
models = ['IDM', 'Krauss', 'Wiedemann', 'RL-Aggressive']
lane_changes = {
    'IDM':            {'Keep': 70, 'Left': 15, 'Right': 15},
    'Krauss':         {'Keep': 60, 'Left': 20, 'Right': 20},
    'Wiedemann':      {'Keep': 65, 'Left': 17, 'Right': 18},
    'RL-Aggressive':  {'Keep': 50, 'Left': 25, 'Right': 25},
}

# === Save predictions to CSV ===
df_export = pd.DataFrame(lane_changes).T.reset_index()
df_export.columns = ['Model', 'Keep', 'Left', 'Right']
df_export['Total Changes'] = df_export['Left'] + df_export['Right']
df_export.to_csv("output/real_model_predictions.csv", index=False)
print("📁 Predictions exported to: output\\real_model_predictions.csv")

# === Pretty Table ===
summary_table = []
for model in models:
    row = lane_changes[model]
    total_changes = row['Left'] + row['Right']
    summary_table.append([model, row['Keep'], row['Left'], row['Right'], total_changes])

print("\n       Lane Change Summary (Balanced Simulation)")
print(tabulate(summary_table, headers=["Model", "Keep", "Left", "Right", "Total Changes"], tablefmt="fancy_grid"))

# === Bar Chart Comparison ===
labels = ['Keep', 'Left', 'Right']
bar_data = np.array([[lane_changes[model][label] for label in labels] for model in models])
x = np.arange(len(models))

fig, ax = plt.subplots(figsize=(10, 6))
width = 0.2
for i, label in enumerate(labels):
    ax.bar(x + i*width, bar_data[:, i], width, label=label)

ax.set_xlabel('Model')
ax.set_ylabel('Count')
ax.set_title('Lane Change Prediction Distribution')
ax.set_xticks(x + width)
ax.set_xticklabels(models)
ax.legend()
plt.tight_layout()
plt.savefig("output/lane_change_comparison.png")
print("📊 Bar chart saved to: output\\lane_change_comparison.png")

# === Best model heuristic ===
best_model = min(summary_table, key=lambda row: row[4])[0]
print(f"\n🏁 Best Model: {best_model} (fewest unnecessary lane changes)")
