# prediction/visualization.py
import carla
import pygame
import numpy as np
import os
import pandas as pd
from datetime import datetime
import math

class LaneChangeVisualizer:
    # Define colors at the class level
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    BLUE = (0, 0, 255)
    YELLOW = (255, 255, 0)
    
    def __init__(self, world=None, width=1280, height=720):
        # Initialize Pygame
        pygame.init()
        pygame.font.init()
        
        # Create display
        self.display_width = width
        self.display_height = height
        self.display = pygame.display.set_mode((width, height), pygame.HWSURFACE | pygame.DOUBLEBUF)
        pygame.display.set_caption("Lane Change Prediction")
        
        # Initialize fonts
        self.font = pygame.font.SysFont('Arial', 20)
        self.large_font = pygame.font.SysFont('Arial', 30)
        
        # Arrow surface for direction indicators
        self.arrow_size = 30
        self.arrow_left = self._create_arrow_surface('left')
        self.arrow_right = self._create_arrow_surface('right')
        self.arrow_none = self._create_arrow_surface('none')
        
        # Table settings
        self.table_x = 20
        self.table_y = 20
        self.table_width = 400
        self.row_height = 40
        self.col_widths = [80, 100, 100, 100]
        
        # Vehicle data storage
        self.vehicle_data = {}
        
        # CARLA world
        self.world = world
        
        # Camera settings
        if world:
            self.spectator = world.get_spectator()
        else:
            self.spectator = None
            
        self.target_vehicle_index = 0  # Index of the current target vehicle
        self.camera_height = 15.0  # Set camera height for top-down view
        
        # Camera smoothing variables - updated for highway speeds
        self.camera_position = None
        self.camera_rotation = None
        self.smoothing_factor = 0.15  # Increased for faster camera response
        
        # Add damping factors with reduced values for quicker movement
        self.position_velocity = carla.Location(0, 0, 0) if world else None
        self.rotation_velocity = carla.Rotation(0, 0, 0) if world else None
        self.position_damping = 0.5  # Reduced for less damping (quicker response)
        self.rotation_damping = 0.5
        
        # Store reference to all vehicles
        self.all_vehicles = []
        
        # For lane change detection
        self.last_predictions = {}
        
    def _create_arrow_surface(self, direction):
        """Create an arrow surface for direction indicators."""
        size = self.arrow_size
        surface = pygame.Surface((size, size), pygame.SRCALPHA)
        
        if direction == 'left':
            # Left arrow
            pygame.draw.polygon(surface, self.BLUE, [(size, 0), (0, size//2), (size, size)])
        elif direction == 'right':
            # Right arrow
            pygame.draw.polygon(surface, self.BLUE, [(0, 0), (size, size//2), (0, size)])
        else:
            # No direction (circle)
            pygame.draw.circle(surface, self.GREEN, (size//2, size//2), size//3)
            
        return surface
    
    def update_vehicle_data(self, vehicle_id, speed, lane_change_pred):
        """Update vehicle data for the visualization."""
        # Store the previous prediction to detect changes
        previous_pred = None
        if vehicle_id in self.vehicle_data:
            previous_pred = self.vehicle_data[vehicle_id].get('lane_change_pred')
        
        # Update the data
        self.vehicle_data[vehicle_id] = {
            'speed': speed,
            'lane_change_pred': lane_change_pred,
            'timestamp': datetime.now()
        }
        
        # Remove old entries (older than 5 seconds)
        current_time = datetime.now()
        to_remove = []
        for vid in self.vehicle_data:
            time_diff = (current_time - self.vehicle_data[vid]['timestamp']).total_seconds()
            if time_diff > 5:
                to_remove.append(vid)
                
        for vid in to_remove:
            del self.vehicle_data[vid]
    
    def update_camera(self, vehicles):
        """Update the spectator camera position to follow the current target vehicle with quick movement for highways."""
        if not vehicles or not self.world or not self.spectator:
            return
        
        # Update the list of all vehicles
        self.all_vehicles = [v for v in vehicles if v.is_alive]
        
        if not self.all_vehicles:
            return
            
        # Ensure the target index is valid
        if self.target_vehicle_index >= len(self.all_vehicles):
            self.target_vehicle_index = 0
        
        # Get the current target vehicle
        target_vehicle = self.all_vehicles[self.target_vehicle_index]
        
        # Get vehicle transform
        vehicle_transform = target_vehicle.get_transform()
        
        # Calculate target camera position - always keep top-down view
        target_location = vehicle_transform.location + carla.Location(z=self.camera_height)
        
        # For highways, add a slight forward offset to look ahead of the vehicle
        # Get forward vector of the vehicle
        forward_vector = vehicle_transform.get_forward_vector()
        # Add a forward offset (look ahead of the vehicle)
        forward_offset = 10.0  # 10 meters ahead
        target_location += carla.Location(
            x=forward_vector.x * forward_offset,
            y=forward_vector.y * forward_offset
        )
        
        # Ensure camera is always pointing straight down (top-down view)
        target_rotation = carla.Rotation(pitch=-90, yaw=vehicle_transform.rotation.yaw)
        
        # Initialize camera position if it's the first update
        if self.camera_position is None:
            self.camera_position = target_location
            self.camera_rotation = target_rotation
            self.position_velocity = carla.Location(0, 0, 0)
            self.rotation_velocity = carla.Rotation(0, 0, 0)
            return
        
        # For fast highway vehicles, use a quicker response if the vehicle is far from camera
        distance_to_target = self.camera_position.distance(target_location)
        adaptive_smoothing = min(1.0, max(self.smoothing_factor, distance_to_target / 30.0))
        
        # Calculate the difference (spring force)
        location_diff = carla.Location(
            target_location.x - self.camera_position.x,
            target_location.y - self.camera_position.y,
            target_location.z - self.camera_position.z
        )
        
        # Update velocity with spring force and damping
        self.position_velocity.x = self.position_velocity.x * self.position_damping + location_diff.x * adaptive_smoothing
        self.position_velocity.y = self.position_velocity.y * self.position_damping + location_diff.y * adaptive_smoothing
        self.position_velocity.z = self.position_velocity.z * self.position_damping + location_diff.z * adaptive_smoothing
        
        # Update position
        self.camera_position.x += self.position_velocity.x
        self.camera_position.y += self.position_velocity.y
        self.camera_position.z += self.position_velocity.z
        
        # Handle yaw wrap-around for smooth rotation
        yaw_diff = (target_rotation.yaw - self.camera_rotation.yaw + 180) % 360 - 180
        
        # Update rotation velocity with spring force and damping
        self.rotation_velocity.yaw = self.rotation_velocity.yaw * self.rotation_damping + yaw_diff * adaptive_smoothing
        
        # Update rotation (keeping pitch at -90)
        self.camera_rotation.yaw = (self.camera_rotation.yaw + self.rotation_velocity.yaw) % 360
        self.camera_rotation.pitch = -90  # Always keep pitch at -90 for top-down view
        self.camera_rotation.roll = 0     # Always keep roll at 0
        
        # Create and set the new camera transform
        camera_transform = carla.Transform(self.camera_position, self.camera_rotation)
        self.spectator.set_transform(camera_transform)
    
    def switch_target_vehicle(self):
        """Switch to the next vehicle in the list."""
        if not self.all_vehicles:
            return
            
        # Increment the target vehicle index
        self.target_vehicle_index = (self.target_vehicle_index + 1) % len(self.all_vehicles)
        
        # Get the current target vehicle ID
        target_id = self.all_vehicles[self.target_vehicle_index].id
        print(f"Switched to vehicle ID: {target_id}")
    
    def render(self):
        """Render the visualization."""
        # Clear the display
        self.display.fill(self.BLACK)
        
        # Render table header
        self._render_table_header()
        
        # Render vehicle data
        self._render_vehicle_data()
        
        # Render timestamp
        timestamp_text = self.font.render(f"Time: {datetime.now().strftime('%H:%M:%S')}", True, self.WHITE)
        self.display.blit(timestamp_text, (self.display_width - 180, 20))
        
        # Render current target vehicle info
        if self.all_vehicles and len(self.all_vehicles) > self.target_vehicle_index:
            target_id = self.all_vehicles[self.target_vehicle_index].id
            vehicle_text = self.font.render(
                f"Following Vehicle ID: {target_id} (Press N to switch)", 
                True, self.WHITE
            )
            self.display.blit(vehicle_text, (self.display_width - 350, 50))
        
        # Update the display
        pygame.display.flip()
    
    def _render_table_header(self):
        """Render the table header."""
        # Table background
        table_height = (len(self.vehicle_data) + 1) * self.row_height
        pygame.draw.rect(self.display, (50, 50, 50), 
                         (self.table_x, self.table_y, self.table_width, table_height))
        
        # Table border
        pygame.draw.rect(self.display, self.WHITE, 
                         (self.table_x, self.table_y, self.table_width, table_height), 2)
        
        # Header row
        x = self.table_x
        y = self.table_y
        headers = ["Vehicle", "Speed", "Prediction", "Direction"]
        
        for i, header in enumerate(headers):
            cell_width = self.col_widths[i]
            header_text = self.font.render(header, True, self.WHITE)
            text_rect = header_text.get_rect(center=(x + cell_width/2, y + self.row_height/2))
            self.display.blit(header_text, text_rect)
            
            # Draw column separator
            if i < len(headers) - 1:
                pygame.draw.line(self.display, self.WHITE, 
                                (x + cell_width, y), 
                                (x + cell_width, y + table_height))
            
            x += cell_width
        
        # Draw header separator
        pygame.draw.line(self.display, self.WHITE, 
                        (self.table_x, y + self.row_height), 
                        (self.table_x + self.table_width, y + self.row_height))
    
    def _render_vehicle_data(self):
        """Render the vehicle data rows."""
        # Sort vehicles by ID for consistent display
        sorted_vehicles = sorted(self.vehicle_data.items())
        
        for i, (vehicle_id, data) in enumerate(sorted_vehicles):
            row = i + 1  # Skip header row
            y = self.table_y + row * self.row_height
            
            # Row background alternating colors
            bg_color = (40, 40, 40) if i % 2 == 0 else (30, 30, 30)
            
            # Highlight the currently followed vehicle
            if self.all_vehicles and len(self.all_vehicles) > self.target_vehicle_index:
                if vehicle_id == self.all_vehicles[self.target_vehicle_index].id:
                    bg_color = (60, 60, 100)  # Highlight color
            
            # Highlight vehicles that are changing lanes
            if data['lane_change_pred'] != 0:  # Not keeping lane
                # Use a softer highlight for lane-changing vehicles
                bg_color = tuple(min(c + 30, 255) for c in bg_color)
            
            pygame.draw.rect(self.display, bg_color, 
                             (self.table_x, y, self.table_width, self.row_height))
            
            # Vehicle ID
            x = self.table_x
            id_text = self.font.render(str(vehicle_id), True, self.WHITE)
            text_rect = id_text.get_rect(center=(x + self.col_widths[0]/2, y + self.row_height/2))
            self.display.blit(id_text, text_rect)
            
            # Speed
            x += self.col_widths[0]
            speed_text = self.font.render(f"{data['speed']:.1f} km/h", True, self.WHITE)
            text_rect = speed_text.get_rect(center=(x + self.col_widths[1]/2, y + self.row_height/2))
            self.display.blit(speed_text, text_rect)
            
            # Prediction text
            x += self.col_widths[1]
            pred_text = ""
            if data['lane_change_pred'] == -1:
                pred_text = "Left"
            elif data['lane_change_pred'] == 1:
                pred_text = "Right"
            else:
                pred_text = "Stay"
                
            prediction_text = self.font.render(pred_text, True, self.WHITE)
            text_rect = prediction_text.get_rect(center=(x + self.col_widths[2]/2, y + self.row_height/2))
            self.display.blit(prediction_text, text_rect)
            
            # Direction arrow
            x += self.col_widths[2]
            if data['lane_change_pred'] == -1:
                arrow = self.arrow_left
            elif data['lane_change_pred'] == 1:
                arrow = self.arrow_right
            else:
                arrow = self.arrow_none
                
            arrow_rect = arrow.get_rect(center=(x + self.col_widths[3]/2, y + self.row_height/2))
            self.display.blit(arrow, arrow_rect)
    
    def handle_events(self):
        """Handle pygame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                elif event.key == pygame.K_n:
                    # Switch target vehicle when 'N' is pressed
                    self.switch_target_vehicle()
        return True
    
    def quit(self):
        """Clean up pygame resources."""
        pygame.quit()