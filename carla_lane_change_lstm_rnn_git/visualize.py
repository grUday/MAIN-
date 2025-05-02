import carla
import matplotlib.pyplot as plt
import json
import time

# Connect to CARLA
client = carla.Client('localhost', 2000)
client.set_timeout(5.0)
world = client.get_world()

# Load vehicle positions from CARLA
def get_vehicle_positions():
    vehicle_list = world.get_actors().filter('vehicle.*')
    x = [v.get_location().x for v in vehicle_list]
    y = [v.get_location().y for v in vehicle_list]
    return x, y

# Load lane change predictions from JSON file
def load_predictions():
    try:
        with open("predictions.json", "r") as f:
            predictions = json.load(f)
        return predictions
    except FileNotFoundError:
        print("⚠ No prediction file found. Run rnn_model.py first!")
        return None

# Real-time visualization loop
plt.ion()  # Interactive mode ON
fig, ax = plt.subplots()

while True:
    ax.clear()  # Clear previous frame

    # Get vehicle positions
    x, y = get_vehicle_positions()

    # Load predictions
    predictions = load_predictions()
    
    # Plot vehicle positions
    ax.scatter(x, y, color='blue', label="Vehicles")

    # If predictions exist, annotate vehicles
    if predictions:
        for i, (px, py) in enumerate(zip(x, y)):
            pred_class = predictions.get(str(i), "Unknown")
            ax.text(px, py, f"{pred_class}", fontsize=10, color="red")

    # Set plot labels
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.set_title("Real-Time Lane Prediction in Digital Twin")
    ax.legend()

    plt.pause(1)  # Update every second
