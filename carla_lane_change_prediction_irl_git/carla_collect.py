import carla
import time
import joblib
import numpy as np
import pandas as pd
from rich.live import Live
from rich.table import Table
from agents.navigation.basic_agent import BasicAgent

# Load IRL reward weights
theta = joblib.load("models/irl_reward_weights.pkl")

# Connect to CARLA
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.load_world('Town04')
blueprint_lib = world.get_blueprint_library()
map = world.get_map()

spawn_point = map.get_spawn_points()[0]
vehicle_bp = blueprint_lib.filter('vehicle.audi.a2')[0]
vehicles, agents = [], []

# Spawn vehicles
for i in range(5):
    offset = carla.Location(x=i * 4.0)
    spawn_tf = carla.Transform(spawn_point.location + offset, spawn_point.rotation)
    vehicle = world.spawn_actor(vehicle_bp, spawn_tf)
    vehicle.set_autopilot(True)
    vehicles.append(vehicle)

    if i == 1 or i == 3:
        agent = BasicAgent(vehicle)
        current_wp = map.get_waypoint(vehicle.get_location())
        next_wp = current_wp.get_right_lane() or current_wp.get_left_lane()
        if next_wp:
            destination = next_wp.transform.location + carla.Location(x=40)
            agent.set_destination(destination)
            agents.append(agent)
        else:
            agents.append(None)
    else:
        agents.append(None)

spectator = world.get_spectator()
previous_states = [None] * len(vehicles)

# Helper to map IRL reward to action
def predict_action(dx, dy, speed, lane_diff):
    features = np.array([dx, dy, speed, lane_diff])
    reward = np.dot(features, theta)
    if reward > 0.2:
        return 1  # right
    elif reward < -0.2:
        return -1  # left
    else:
        return 0  # no change

def direction_arrow(a):
    return "←" if a == -1 else "→" if a == 1 else "↑"

def create_table(frame_count, rows):
    table = Table(title=f"🧠 IRL Lane Change Predictions – Frame {frame_count}")
    table.add_column("ID", justify="right")
    table.add_column("X", justify="right")
    table.add_column("Y", justify="right")
    table.add_column("Speed", justify="right")
    table.add_column("Lane ID", justify="center")
    table.add_column("IRL Prediction", justify="center")
    for row in rows:
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
                lane_id = map.get_waypoint(tf.location).lane_id
                state = {
                    "x": tf.location.x,
                    "y": tf.location.y,
                    "vx": vel.x,
                    "vy": vel.y,
                    "laneId": lane_id
                }

                # Predict IRL action using change from previous state
                if previous_states[i] is not None:
                    dx = state["x"] - previous_states[i]["x"]
                    dy = state["y"] - previous_states[i]["y"]
                    lane_diff = state["laneId"] - previous_states[i]["laneId"]
                    a_pred = predict_action(dx, dy, speed, lane_diff)
                    arrow = direction_arrow(a_pred)
                else:
                    arrow = "..."

                previous_states[i] = state

                table_rows.append([
                    str(i),
                    f"{state['x']:.2f}",
                    f"{state['y']:.2f}",
                    f"{speed:.2f}",
                    str(lane_id),
                    arrow
                ])

                if agents[i]:
                    control = agents[i].run_step()
                    vehicle.apply_control(control)

                if i == 0:
                    cam_loc = tf.location + carla.Location(z=50)
                    spectator.set_transform(carla.Transform(cam_loc, carla.Rotation(pitch=-90)))

            live.update(create_table(frame, table_rows))

except KeyboardInterrupt:
    print("\n⛔ Stopped by user.")
finally:
    for v in vehicles:
        v.destroy()
    print("✅ Cleaned up all vehicles.")
