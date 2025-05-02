# data_collection/config.py
import carla

class DataCollectionConfig:
    # CARLA settings
    HOST = 'localhost'
    PORT = 2000
    TOWN = 'Town05'  # Good for lane changes
    NUM_VEHICLES = 20
    WEATHER = carla.WeatherParameters.ClearNoon
    
    # Data collection settings
    FRAMES_TO_RECORD = 10000
    FRAME_SKIP = 3  # Record every n-th frame
    
    # Feature extraction settings
    HISTORY_SECONDS = 3.0  # Amount of history to consider for prediction
    PREDICTION_HORIZON = 2.0  # How far ahead to predict (in seconds)
    
    # Save settings
    OUTPUT_PATH = '../data/raw/traffic_data.csv'