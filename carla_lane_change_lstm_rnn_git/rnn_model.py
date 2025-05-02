import carla
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time

# ✅ Connect to CARLA
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.get_world()

# ✅ Force world update to refresh vehicle states
world.tick()
time.sleep(2)  # Allow physics engine to update

# ✅ Get all vehicles and assign names
vehicles = world.get_actors().filter('vehicle.*')
vehicle_names = {vehicle.id: f"Vehicle {i+1}" for i, vehicle in enumerate(vehicles)}

if not vehicles:
    print("⚠ No vehicles found in CARLA! Exiting...")
    exit()

print(f"🚗 Found {len(vehicles)} vehicles!")

# ✅ Ensure vehicles are moving before retrieving velocity
for vehicle in vehicles:
    vehicle.apply_control(carla.VehicleControl(throttle=0.3, steer=0.0))

time.sleep(2)  # Allow movement

# ✅ Define LSTM Lane Change Model
class LaneChangeModel(nn.Module):
    def __init__(self):
        super(LaneChangeModel, self).__init__()
        self.lstm = nn.LSTM(input_size=4, hidden_size=32, num_layers=2, batch_first=True)
        self.fc = nn.Linear(32, 3)  # Output: Left, Keep, Right

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        return self.fc(hidden[-1])

# ✅ Load trained model
model = LaneChangeModel()
model.load_state_dict(torch.load("lane_change_model.pth"))
model.eval()

# ✅ Collect vehicle data (Position & Velocity)
def collect_vehicle_data(vehicle):
    if vehicle is None or not vehicle.is_alive:
        return [0.0, 0.0, 0.0, 0.0]

    loc = vehicle.get_location()

    world.tick()
    time.sleep(0.5)  # Ensure updated velocity

    velocity = vehicle.get_velocity()
    vx, vy = round(velocity.x, 2), round(velocity.y, 2)

    # Retry if velocity is too low
    if abs(vx) < 0.1 and abs(vy) < 0.1:
        print(f"⚠ {vehicle.id} has low velocity ({vx}, {vy}). Retrying...")
        time.sleep(2)
        velocity = vehicle.get_velocity()
        vx, vy = round(velocity.x, 2), round(velocity.y, 2)

    return [round(loc.x, 2), round(loc.y, 2), vx, vy]

# ✅ Predict lane change dynamically
def predict_lane_change(vehicle_data):
    vehicle_tensor = torch.tensor([vehicle_data], dtype=torch.float32).unsqueeze(0)
    
    with torch.no_grad():
        output_scores = model(vehicle_tensor)
        probabilities = F.softmax(output_scores / 1.0, dim=1).detach().numpy()[0]  

    predicted_class = np.random.choice([0, 1, 2], p=probabilities)  

    return int(predicted_class), probabilities

# ✅ Arrow graphics for better visualization
arrow_graphics = ["⬅️  Left", "⬆️  Keep Lane", "➡️  Right"]

# ✅ Run prediction for 10 epochs
for epoch in range(10):
    print("\n" + "═" * 50)
    print(f"🔄 𝐄𝐩𝐨𝐜𝐡 {epoch+1}/10")
    print("═" * 50)

    left_count, keep_count, right_count = 0, 0, 0  

    for vehicle in vehicles:
        if vehicle.id not in vehicle_names:
            continue  

        vehicle_name = vehicle_names[vehicle.id]

        world.tick()

        vehicle_data = collect_vehicle_data(vehicle)
        
        if vehicle_data[2] == 0.0 and vehicle_data[3] == 0.0:
            print(f"⚠ {vehicle_name} is stationary. Skipping prediction.")
            continue
        
        predicted_class, probabilities = predict_lane_change(vehicle_data)

        if predicted_class == 0:
            left_count += 1
        elif predicted_class == 1:
            keep_count += 1
        elif predicted_class == 2:
            right_count += 1

        # ✅ Clear and structured output
        print(f"📍 {vehicle_name}  |  📌 Position: ({vehicle_data[0]}, {vehicle_data[1]})  |  🚀 Velocity: ({vehicle_data[2]}, {vehicle_data[3]})")
        print(f"🚗 {vehicle_name}  |  🛣 Lane Change: {arrow_graphics[predicted_class]}  (L: {probabilities[0]:.2f}, K: {probabilities[1]:.2f}, R: {probabilities[2]:.2f})")
        print("─" * 50)

    print(f"📊 𝐄𝐩𝐨𝐜𝐡 {epoch+1} 𝐒𝐮𝐦𝐦𝐚𝐫𝐲:  ⬅️ Left={left_count}  |  ⬆️ Keep={keep_count}  |  ➡️ Right={right_count}")  

    time.sleep(1)

print("\n🛑 𝐋𝐚𝐧𝐞 𝐂𝐡𝐚𝐧𝐠𝐞 𝐏𝐫𝐞𝐝𝐢𝐜𝐭𝐢𝐨𝐧 𝐂𝐨𝐦𝐩𝐥𝐞𝐭𝐞𝐝! 🎉")
