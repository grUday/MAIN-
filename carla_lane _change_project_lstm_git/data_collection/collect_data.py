# data_collection/collect_data.py
import carla
import random
import pandas as pd
import numpy as np
import time
import os
import sys
import argparse
import math
import csv
from tqdm import tqdm

from data_collection.config import DataCollectionConfig as config
from data_collection.utils import get_vehicle_features

def filter_highway_spawn_points(spawn_points, world):
    """Filter spawn points to only include those on highways/main roads."""
    highway_spawn_points = []
    
    # Get the map
    carla_map = world.get_map()
    
    for spawn_point in spawn_points:
        # Get the waypoint for this spawn point
        waypoint = carla_map.get_waypoint(spawn_point.location, 
                                          project_to_road=True,
                                          lane_type=carla.LaneType.Driving)
        
        # Check if this is a highway (based on road width and number of lanes)
        # Highway waypoints typically have:
        # - Multiple lanes
        # - Larger road width
        # - Higher speed limit
        if (waypoint.lane_type == carla.LaneType.Driving and
            waypoint.get_right_lane() is not None and  # Has multiple lanes
            waypoint.lane_width > 3.5):  # Wider lanes
            
            # Further filter for straight sections to make lane changes more predictable
            next_waypoint = waypoint.next(10.0)[0]  # Get waypoint 10m ahead
            if next_waypoint:
                # Calculate angle between current and next waypoint (straight roads have angle close to 0)
                angle = math.degrees(math.atan2(next_waypoint.transform.rotation.yaw - waypoint.transform.rotation.yaw, 
                                               waypoint.transform.location.distance(next_waypoint.transform.location)))
                if abs(angle) < 5.0:  # Relatively straight road
                    highway_spawn_points.append(spawn_point)
    
    # If we didn't find enough highway points, return a subset of the original points
    if len(highway_spawn_points) < 10:
        print("Warning: Not enough highway spawn points found. Using generic spawn points.")
        return random.sample(spawn_points, min(20, len(spawn_points)))
    
    return highway_spawn_points

def main():
    """Main function to collect traffic data from CARLA simulation with focus on highway scenarios."""
    # Connect to CARLA server
    client = carla.Client(config.HOST, config.PORT)
    client.set_timeout(10.0)
    
    # Load desired map - Town04 has good highways
    world = client.load_world('Town04')
    print(f"Loaded world: Town04 (good for highway scenarios)")
    
    # Set up the simulator in synchronous mode
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05  # 20 FPS
    world.apply_settings(settings)
    
    # Set weather
    weather = carla.WeatherParameters(
        cloudiness=10.0,
        precipitation=0.0,
        sun_altitude_angle=70.0,
        precipitation_deposits=0.0,
        wind_intensity=0.0,
        fog_density=0.0,
        wetness=0.0
    )
    world.set_weather(weather)
    
    # Create the ego vehicle and other actors
    blueprint_library = world.get_blueprint_library()
    vehicle_blueprints = blueprint_library.filter('vehicle.*')
    
    # Filter out bicycles and motorcycles which are less likely to change lanes
    vehicle_blueprints = [bp for bp in vehicle_blueprints if int(bp.get_attribute('number_of_wheels')) >= 4]
    
    # Get all spawn points and filter for highway
    all_spawn_points = world.get_map().get_spawn_points()
    highway_spawn_points = filter_highway_spawn_points(all_spawn_points, world)
    
    print(f"Found {len(highway_spawn_points)} highway spawn points out of {len(all_spawn_points)} total spawn points")
    
    # Create traffic manager
    traffic_manager = client.get_trafficmanager(8000)
    traffic_manager.set_synchronous_mode(True)
    traffic_manager.set_global_distance_to_leading_vehicle(4.0)  # Larger distance for highway
    traffic_manager.global_percentage_speed_difference(0.0)  # Normal speed for smooth traffic flow
    
    # Spawn vehicles
    vehicles_list = []
    try:
        print(f"Spawning {config.NUM_VEHICLES} vehicles on highway...")
        
        # Use highway spawn points in sequence to avoid overlap
        for i in range(min(config.NUM_VEHICLES, len(highway_spawn_points))):
            # Choose a random blueprint
            blueprint = random.choice(vehicle_blueprints)
            
            # Try to spawn the vehicle on highway
            spawn_point = highway_spawn_points[i]
            vehicle = world.try_spawn_actor(blueprint, spawn_point)
            
            if vehicle is not None:
                vehicle.set_autopilot(True, traffic_manager.get_port())
                
                # Set driving behavior for more lane changes
                traffic_manager.random_left_lanechange_percentage(vehicle, 70)  # Increased probability
                traffic_manager.random_right_lanechange_percentage(vehicle, 70) # Increased probability
                traffic_manager.vehicle_percentage_speed_difference(vehicle, random.uniform(-15, 10))
                traffic_manager.auto_lane_change(vehicle, True)
                traffic_manager.update_vehicle_lights(vehicle, True)
                
                vehicles_list.append(vehicle)
        
        print(f"Spawned {len(vehicles_list)} vehicles.")
        
        # Let the simulation run for a few seconds to stabilize
        print("Letting vehicles stabilize for a few seconds...")
        for _ in range(100):  # More ticks for better stabilization
            world.tick()
        
        # Convert relative path to absolute path for OUTPUT_PATH
        absolute_output_path = os.path.abspath(os.path.join(os.getcwd(), config.OUTPUT_PATH.replace('../', '')))
        
        # Prepare the output directory with absolute path
        output_dir = os.path.dirname(absolute_output_path)
        print(f"Creating directory: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize data collection
        data = []
        frame_count = 0
        
        # Collect data
        print(f"Starting data collection for {config.FRAMES_TO_RECORD} frames...")
        with tqdm(total=config.FRAMES_TO_RECORD) as pbar:
            while frame_count < config.FRAMES_TO_RECORD:
                # Tick the world
                world.tick()
                
                # Only record every n-th frame
                if frame_count % config.FRAME_SKIP == 0:
                    # Collect data for each vehicle
                    for vehicle in vehicles_list:
                        if vehicle.is_alive:
                            features = get_vehicle_features(vehicle, world)
                            data.append(features)
                
                frame_count += 1
                pbar.update(1)
        
        # Convert data to DataFrame and save to CSV
        print("Processing and saving data...")
        df = pd.DataFrame(data)
        
        # Save to the absolute path
        print(f"Saving to: {absolute_output_path}")
        df.to_csv(absolute_output_path, index=False)
        print(f"Data saved to {absolute_output_path}")
        
        # Also save to the original expected path for compatibility
        original_expected_path = os.path.join(os.getcwd(), 'data/raw/traffic_data.csv')
        os.makedirs(os.path.dirname(original_expected_path), exist_ok=True)
        df.to_csv(original_expected_path, index=False)
        print(f"Backup data saved to {original_expected_path}")
        
    finally:
        # Clean up
        print("Cleaning up...")
        for vehicle in vehicles_list:
            if vehicle.is_alive:
                vehicle.destroy()
        print("Done!")

if __name__ == "__main__":
    main()