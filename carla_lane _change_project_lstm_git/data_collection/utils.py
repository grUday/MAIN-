# data_collection/utils.py
import carla
import random
import numpy as np
import pandas as pd
import time
import math

def get_lane_change_status(vehicle):
    """Determine if a vehicle is changing lanes based on its velocity and direction."""
    velocity = vehicle.get_velocity()
    transform = vehicle.get_transform()
    forward_vector = transform.get_forward_vector()
    
    # Calculate the angle between velocity and forward vector
    velocity_vector = np.array([velocity.x, velocity.y])
    if np.linalg.norm(velocity_vector) < 0.1:  # If almost stationary
        return 0  # No lane change
    
    velocity_norm = velocity_vector / np.linalg.norm(velocity_vector)
    forward_norm = np.array([forward_vector.x, forward_vector.y])
    
    cross_product = np.cross(forward_norm, velocity_norm)
    
    # Determine lane change direction (-1: left, 0: none, 1: right)
    if abs(cross_product) < 0.15:
        return 0  # No lane change
    elif cross_product > 0:
        return 1  # Right lane change
    else:
        return -1  # Left lane change

def get_vehicle_features(vehicle, world):
    """Extract features for a vehicle."""
    # Basic kinematic features
    velocity = vehicle.get_velocity()
    speed = 3.6 * math.sqrt(velocity.x**2 + velocity.y**2)  # km/h
    
    acceleration = vehicle.get_acceleration()
    acc_magnitude = math.sqrt(acceleration.x**2 + acceleration.y**2)
    
    angular_velocity = vehicle.get_angular_velocity()
    
    # Vehicle transform
    transform = vehicle.get_transform()
    location = transform.location
    rotation = transform.rotation
    
    # Lane information
    waypoint = world.get_map().get_waypoint(location)
    lane_id = waypoint.lane_id
    lane_width = waypoint.lane_width
    
    # Lane change status
    lane_change = get_lane_change_status(vehicle)
    
    # Traffic light state
    traffic_light = vehicle.get_traffic_light_state()
    
    # Surrounding vehicles (simplified)
    surrounding_info = {
        'front_dist': 100.0,  # Default large value
        'rear_dist': 100.0,
        'left_dist': 100.0,
        'right_dist': 100.0
    }
    
    actor_list = world.get_actors().filter('vehicle.*')
    vehicle_location = vehicle.get_location()
    vehicle_forward = transform.get_forward_vector()
    
    for actor in actor_list:
        if actor.id != vehicle.id:
            actor_location = actor.get_location()
            distance = vehicle_location.distance(actor_location)
            
            if distance > 50:  # Ignore vehicles too far away
                continue
                
            # Get vector from current vehicle to the other vehicle
            to_actor = np.array([
                actor_location.x - vehicle_location.x,
                actor_location.y - vehicle_location.y
            ])
            to_actor_norm = np.linalg.norm(to_actor)
            if to_actor_norm < 0.01:
                continue
                
            to_actor = to_actor / to_actor_norm
            
            # Forward vector as numpy array
            forward = np.array([vehicle_forward.x, vehicle_forward.y])
            right = np.array([-vehicle_forward.y, vehicle_forward.x])  # 90 degrees clockwise
            
            # Dot products to determine direction
            forward_proj = np.dot(to_actor, forward)
            right_proj = np.dot(to_actor, right)
            
            # Update closest vehicles in each direction
            if forward_proj > 0.7:  # Vehicle is in front
                surrounding_info['front_dist'] = min(surrounding_info['front_dist'], distance)
            elif forward_proj < -0.7:  # Vehicle is behind
                surrounding_info['rear_dist'] = min(surrounding_info['rear_dist'], distance)
            
            if right_proj > 0.7:  # Vehicle is to the right
                surrounding_info['right_dist'] = min(surrounding_info['right_dist'], distance)
            elif right_proj < -0.7:  # Vehicle is to the left
                surrounding_info['left_dist'] = min(surrounding_info['left_dist'], distance)
    
    # Compile all features
    features = {
        'vehicle_id': vehicle.id,
        'timestamp': time.time(),
        'position_x': location.x,
        'position_y': location.y,
        'position_z': location.z,
        'rotation_pitch': rotation.pitch,
        'rotation_yaw': rotation.yaw,
        'rotation_roll': rotation.roll,
        'velocity_x': velocity.x,
        'velocity_y': velocity.y,
        'speed': speed,
        'acceleration_x': acceleration.x,
        'acceleration_y': acceleration.y,
        'acceleration_magnitude': acc_magnitude,
        'angular_velocity_z': angular_velocity.z,  # Yaw rate
        'lane_id': lane_id,
        'lane_width': lane_width,
        'lane_change': lane_change,
        'traffic_light_state': str(traffic_light),
        'front_vehicle_dist': surrounding_info['front_dist'],
        'rear_vehicle_dist': surrounding_info['rear_dist'],
        'left_vehicle_dist': surrounding_info['left_dist'],
        'right_vehicle_dist': surrounding_info['right_dist']
    }
    
    return features