import carla

client = carla.Client("localhost", 2000)
client.set_timeout(5.0)
world = client.get_world()
spectator = world.get_spectator()
transform = spectator.get_transform()

print("🚀 Camera Location:")
print(f"Location: {transform.location}")
print(f"Rotation: {transform.rotation}")
