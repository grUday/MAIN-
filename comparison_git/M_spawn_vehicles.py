import carla
import pickle
import time

MODEL_PICKLE_PATH = "models/Intelligent_Driver_Model_(IDM).pickle"
MAP_NAME = "Town04"
SPAWN_Z = 1.2

def main():
    client = carla.Client("localhost", 2000)
    client.set_timeout(30.0)

    print(f"🌍 Loading map: {MAP_NAME}...")
    world = client.load_world(MAP_NAME)
    time.sleep(2.0)

    print("🛑 Removing existing vehicles...")
    actors = world.get_actors().filter("vehicle.*")
    for actor in actors:
        actor.destroy()
    print("✅ Previous vehicles removed.")

    print(f"📦 Loading vehicle data from: {MODEL_PICKLE_PATH}")
    with open(MODEL_PICKLE_PATH, "rb") as f:
        vehicle_data = pickle.load(f)
    print("✅ Vehicle data loaded successfully!")

    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.filter("vehicle.tesla.model3")[0]

    spawned_vehicles = []
    for idx, row in vehicle_data.iterrows():
        x, y = row["x"], row["y"]
        transform = carla.Transform(carla.Location(x=x, y=y, z=SPAWN_Z))
        vehicle = world.try_spawn_actor(vehicle_bp, transform)
        if vehicle:
            print(f"✅ Vehicle {idx} spawned at x={x:.2f}, y={y:.2f}")
            spawned_vehicles.append(vehicle)
        else:
            print(f"❌ Vehicle {idx} spawn failed due to collision.")
        time.sleep(0.3)

    if spawned_vehicles:
        spectator = world.get_spectator()
        vehicle_location = spawned_vehicles[0].get_location()
        camera_transform = carla.Transform(vehicle_location + carla.Location(z=40), carla.Rotation(pitch=-90))
        spectator.set_transform(camera_transform)
        print("🎥 Moved camera to spawned vehicles.")
    else:
        print("⚠️ No vehicles spawned.")

    print(f"🚗 Total vehicles spawned: {len(spawned_vehicles)}")

if __name__ == "__main__":
    main()
