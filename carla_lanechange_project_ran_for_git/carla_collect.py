import carla
import time
import joblib
import numpy as np
import pandas as pd
from rich.live import Live
from rich.table import Table
from agents.navigation.basic_agent import BasicAgent  # Comes with CARLA PythonAPI

# Load model
model = joblib.load("models/lane_change_model.pkl")

# Connect to CARLA
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.load_world('Town04')  # Town04 has highway-style roads
blueprint_lib = world.get_blueprint_library()
map = world.get_map()

spawn_point = map.get_spawn_points()[0]
vehicle_bp = blueprint_lib.filter('vehicle.audi.a2')[0]
vehicles = []
agents = []

# Spawn 5 vehicles
for i in range(5):
    offset = carla.Location(x=i * 4.0)
    spawn_tf = carla.Transform(spawn_point.location + offset, spawn_point.rotation)
    vehicle = world.spawn_actor(vehicle_bp, spawn_tf)
    vehicle.set_autopilot(True)  # default
    vehicles.append(vehicle)

    if i == 1 or i == 3:
        # Use BasicAgent for lane change behavior
        agent = BasicAgent(vehicle)
        current_wp = map.get_waypoint(vehicle.get_location())

        # Try right lane first, if available
        next_wp = current_wp.get_right_lane() or current_wp.get_left_lane()
        if next_wp:
            destination = next_wp.transform.location + carla.Location(x=40)
            agent.set_destination(destination)
            agents.append(agent)
        else:
            agents.append(None)
    else:
        agents.append(None)

# Set up camera
spectator = world.get_spectator()

def direction_arrow(pred):
    return "←" if pred == -1 else "→" if pred == 1 else "↑"

def create_table(frame_count, data):
    table = Table(title=f"🚗 Lane Change Prediction Dashboard - Frame {frame_count}")
    table.add_column("ID", justify="right")
    table.add_column("X", justify="right")
    table.add_column("Y", justify="right")
    table.add_column("Speed", justify="right")
    table.add_column("Lane ID", justify="center")
    table.add_column("Prediction", justify="center")

    for row in data:
        table.add_row(*row)
    return table

try:
    frame = 0
    with Live(refresh_per_second=4) as live:
        while True:
            world.tick()
            time.sleep(0.1)
            frame += 1

            table_rows = []

            for i, vehicle in enumerate(vehicles):
                tf = vehicle.get_transform()
                vel = vehicle.get_velocity()
                speed = np.linalg.norm([vel.x, vel.y])
                dx, dy = vel.x * 0.1, vel.y * 0.1
                lane_id = map.get_waypoint(tf.location).lane_id

                # Predict with proper feature names
                features = pd.DataFrame([[dx, dy, speed]], columns=["dx", "dy", "speed"])
                pred = model.predict(features)[0]

                table_rows.append([
                    str(i),
                    f"{tf.location.x:.2f}",
                    f"{tf.location.y:.2f}",
                    f"{speed:.2f}",
                    str(lane_id),
                    direction_arrow(pred)
                ])

                # Camera follows vehicle 0
                if i == 0:
                    cam_loc = tf.location + carla.Location(z=50)
                    spectator.set_transform(carla.Transform(cam_loc, carla.Rotation(pitch=-90)))

                # If it's a controlled agent, override autopilot
                if agents[i]:
                    control = agents[i].run_step()
                    vehicle.apply_control(control)

            live.update(create_table(frame, table_rows))

except KeyboardInterrupt:
    print("\n⛔ Stopped by user.")

finally:
    for v in vehicles:
        v.destroy()
    print("✅ All vehicles cleaned up.")
