import os
import subprocess
import time

# ---------- CONFIG ----------
carla_path = "C:/carla_lane_change_lstm_rnn/lane-change-prediction-lstm-master/carla/CarlaUE4.exe"  # Update this path as needed!
town_name = "Town04"
frame_wait = 20  # Seconds to wait for CARLA to load

# ---------- STEP 0: Launch CARLA ----------
print(f"🚀 Launching CARLA ({town_name})...")
carla_proc = subprocess.Popen([carla_path, "-carla-server", "-quality-level=Low", f"-world-port=2000", f"-benchmark", f"-fps=20"])

print(f"⌛ Waiting {frame_wait}s for CARLA to initialize...")
time.sleep(frame_wait)

# ---------- STEP 1: Collect expert driving data ----------
print("\n🎥 Running expert simulation...")
subprocess.run(["python", "carla_collect.py"])

# ---------- STEP 2: Process data into features ----------
print("\n🧮 Processing trajectory into features...")
subprocess.run(["python", "process_data.py"])

# ---------- STEP 3: Train IRL model ----------
print("\n🧠 Training IRL reward function...")
subprocess.run(["python", "train_irl.py"])

# ---------- STEP 4: Live IRL lane change predictions ----------
print("\n📺 Starting live IRL prediction dashboard...")
subprocess.run(["python", "carla_collect.py"])

# ---------- Clean up ----------
print("\n🧹 All done! Press Ctrl+C if you want to stop CARLA manually.")
