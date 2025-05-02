import carla
import time
import joblib
import numpy as np
import pandas as pd
from rich.live import Live
from rich.table import Table
from agents.navigation.basic_agent import BasicAgent
import pygame

# Load lane change model
model = joblib.load("models/lane_change_model.pkl")

# Connect to CARLA
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.load_world('Town04')
blueprint_lib = world.get_blueprint_library()
map = world.get_map()

vehicle_bp = blueprint_lib.filter('vehicle.audi.a2')[0]
spawn_points = map.get_spawn_points()
vehicles = []
agents = []
prev_velocities = []

# Choose non-overlapping spawn points across lanes
used_indices = [0, 10, 20, 30, 40]

for i, idx in enumerate(used_indices):
    spawn_tf = spawn_points[idx]
    vehicle = world.try_spawn_actor(vehicle_bp, spawn_tf)

    if not vehicle:
        print(f"❌ Could not spawn vehicle {i} at spawn point {idx}")
        continue

    vehicles.append(vehicle)
    prev_velocities.append(vehicle.get_velocity())

    if i == 1 or i == 3:
        # Controlled with lane change destination
        agent = BasicAgent(vehicle)
        current_wp = map.get_waypoint(vehicle.get_location())
        next_wp = current_wp.get_right_lane() or current_wp.get_left_lane()
        if next_wp:
            destination = next_wp.transform.location + carla.Location(x=50)
            agent.set_destination(destination)
        agents.append(agent)
        vehicle.set_autopilot(False)
    else:
        agents.append(None)
        vehicle.set_autopilot(True)

# Setup spectator camera
spectator = world.get_spectator()
current_view = "driver"

# Interpolation helpers
def interpolate_loc(loc1, loc2, alpha):
    return carla.Location(
        x=loc1.x + (loc2.x - loc1.x) * alpha,
        y=loc1.y + (loc2.y - loc1.y) * alpha,
        z=loc1.z + (loc2.z - loc1.z) * alpha
    )

def interpolate_rot(rot1, rot2, alpha):
    return carla.Rotation(
        pitch=rot1.pitch + (rot2.pitch - rot1.pitch) * alpha,
        yaw=rot1.yaw + (rot2.yaw - rot1.yaw) * alpha,
        roll=rot1.roll + (rot2.roll - rot1.roll) * alpha
    )

# Utility functions
def direction_arrow(pred):
    return "←" if pred == -1 else "→" if pred == 1 else "↑"

def create_table(frame_count, data):
    table = Table(title=f"🚗 Lane Change Dashboard - Frame {frame_count}")
    table.add_column("ID", justify="right")
    table.add_column("Speed", justify="right")
    table.add_column("Acc", justify="right")
    table.add_column("Yaw", justify="right")
    table.add_column("Lane", justify="center")
    table.add_column("Prediction", justify="center")
    for row in data:
        table.add_row(*row)
    return table

# Setup key input
pygame.init()
pygame.display.set_mode((200, 100))  # Needed to capture keys

try:
    frame = 0
    prev_cam_loc = spectator.get_transform().location
    prev_cam_rot = spectator.get_transform().rotation

    with Live(refresh_per_second=4) as live:
        while True:
            world.tick()
            time.sleep(0.1)
            frame += 1

            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN and event.key == pygame.K_TAB:
                    current_view = "top" if current_view == "driver" else "driver"
                    print(f"🔁 Switched view to: {current_view}")

            table_rows = []

            for i, vehicle in enumerate(vehicles):
                tf = vehicle.get_transform()
                vel = vehicle.get_velocity()
                speed = np.linalg.norm([vel.x, vel.y])

                # Acceleration from Δv
                prev_vel = prev_velocities[i]
                acc_x = (vel.x - prev_vel.x) / 0.1
                acc_y = (vel.y - prev_vel.y) / 0.1
                acc = np.linalg.norm([acc_x, acc_y])
                prev_velocities[i] = vel

                yaw = tf.rotation.yaw
                lane_id = map.get_waypoint(tf.location).lane_id

                features = pd.DataFrame([[vel.x * 0.1, vel.y * 0.1, speed]], columns=["dx", "dy", "speed"])
                pred = model.predict(features)[0]
                arrow = direction_arrow(pred)

                info_text = f"{arrow} | {speed:.1f} m/s\na: {acc:.1f} m/s² | θ: {yaw:.0f}°"
                world.debug.draw_string(
                    tf.location + carla.Location(z=2.5),
                    info_text,
                    draw_shadow=True,
                    color=carla.Color(255, 255, 0),
                    life_time=0.2
                )

                table_rows.append([
                    str(i),
                    f"{speed:.1f}",
                    f"{acc:.1f}",
                    f"{yaw:.0f}",
                    str(lane_id),
                    arrow
                ])

                # Camera follow
                if i == 0:
                    if current_view == "driver":
                        target_loc = tf.location + carla.Location(x=0.5, z=1.7)  # Driver's seat view
                        target_rot = tf.rotation
                    else:
                        target_loc = tf.location + carla.Location(z=15)  # Top-down view
                        target_rot = carla.Rotation(pitch=-90, yaw=tf.rotation.yaw)

                    # Adjust interpolation speed for smoother camera movement
                    interpolation_factor = 0.1  # Lower values result in smoother movement

                    # Interpolate location and rotation
                    cam_loc = interpolate_loc(prev_cam_loc, target_loc, interpolation_factor)
                    cam_rot = interpolate_rot(prev_cam_rot, target_rot, interpolation_factor)

                    # Update spectator camera
                    spectator.set_transform(carla.Transform(cam_loc, cam_rot))

                    prev_cam_loc = cam_loc
                    prev_cam_rot = cam_rot

                # Agent movement
                if agents[i]:
                    control = agents[i].run_step()
                    vehicle.apply_control(control)

            live.update(create_table(frame, table_rows))

except KeyboardInterrupt:
    print("⛔ Simulation stopped by user.")

finally:
    for v in vehicles:
        v.destroy()
    pygame.quit()
    print("✅ All vehicles cleaned up.")
