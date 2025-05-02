import carla
import json

# Connect to CARLA
client = carla.Client('localhost', 2000)
client.set_timeout(5.0)  # Ensure connection timeout

# Get the world
world = client.get_world()

# Buffer to store collected data points
vehicle_data_buffer = []

# Function to collect vehicle data
def collect_vehicle_data():
    global vehicle_data_buffer
    vehicle_list = world.get_actors().filter('vehicle.*')  # Get all vehicles

    for vehicle in vehicle_list:
        loc = vehicle.get_location()
        velocity = vehicle.get_velocity()

        # Append new data point (x, y, vx, vy)
        vehicle_data_buffer.append([loc.x, loc.y, velocity.x, velocity.y])

        # Keep last 25 points
        if len(vehicle_data_buffer) > 25:
            vehicle_data_buffer.pop(0)

    return vehicle_data_buffer

# Save collected data to a JSON file
def save_data_to_json():
    with open("vehicle_data.json", "w") as f:
        json.dump(vehicle_data_buffer, f)

# Run the data collection process
print("📡 Collecting Real-Time Vehicle Data from CARLA...\n")

for _ in range(50):  # Collect for 50 cycles (modify as needed)
    data = collect_vehicle_data()
    save_data_to_json()
    print(f"Collected {len(data)} data points.")
