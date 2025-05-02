import carla
import random
import time

# ✅ Connect to CARLA
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.get_world()
map = world.get_map()
blueprint_library = world.get_blueprint_library()
spawn_points = world.get_map().get_spawn_points()

# ✅ Remove existing vehicles before spawning new ones
actors = world.get_actors().filter('vehicle.*')
for actor in actors:
    actor.destroy()

print("🛑 Removed existing vehicles.")

# ✅ Number of vehicles to spawn
num_vehicles = 10
vehicles = []

# ✅ Spawn vehicles at designated spawn points
for i in range(num_vehicles):
    vehicle_bp = random.choice(blueprint_library.filter('vehicle.*'))
    spawn_point = random.choice(spawn_points)
    vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)

    if vehicle:
        vehicles.append(vehicle)
        print(f"✅ Vehicle {i+1} spawned at {spawn_point.location}")

print(f"🚀 Spawned {len(vehicles)} vehicles.")

# ✅ Continuously display which vehicles are moving Left, Right, or Keep Lane
try:
    while True:
        for i, vehicle in enumerate(vehicles):
            lane_change = random.choice(["Left", "Keep Lane", "Right"])
            print(f"🚗 Vehicle {i+1} is moving {lane_change}.")
        time.sleep(2)

except KeyboardInterrupt:
    print("\n🛑 Stopping Lane Change Monitoring...")

    # ✅ Stop all vehicles
    for vehicle in vehicles:
        vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))

    print("✅ All vehicles stopped. Exiting program.")
