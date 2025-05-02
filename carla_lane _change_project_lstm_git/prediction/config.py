# prediction/config.py
import carla

class PredictionConfig:
    # CARLA settings
    HOST = 'localhost'
    PORT = 2000
    TOWN = 'Town05'
    NUM_VEHICLES = 20
    WEATHER = carla.WeatherParameters.ClearNoon
    
    # Visualization settings
    DISPLAY_WIDTH = 1280
    DISPLAY_HEIGHT = 720
    CAMERA_HEIGHT = 15.0  # Height of the spectator camera above vehicles
    
    # Prediction settings
    SEQUENCE_LENGTH = 30  # Must match the sequence length used in training
    PREDICTION_INTERVAL = 10  # Make predictions every N frames
    
    # Run settings
    SIMULATION_SECONDS = 60  # How long to run the simulation