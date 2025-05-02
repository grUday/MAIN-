# data_processing/utils.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os

def create_sequences(data, time_steps, target_column='lane_change'):
    """
    Create input sequences and target values for time series prediction.
    
    Args:
        data (DataFrame): Data frame with sorted time series data
        time_steps (int): Number of time steps to consider for each sequence
        target_column (str): Column name for the target variable
        
    Returns:
        X (np.array): Feature sequences of shape (n_samples, time_steps, n_features)
        y (np.array): Target values of shape (n_samples,)
    """
    sequences = []
    targets = []
    
    # Group data by vehicle ID
    grouped = data.groupby('vehicle_id')
    
    for vehicle_id, group in grouped:
        # Sort by timestamp
        group = group.sort_values('timestamp')
        
        # Extract feature columns (exclude timestamp, vehicle_id, and target)
        features = group.drop(['timestamp', 'vehicle_id', target_column], axis=1).values
        
        # Extract target column
        target = group[target_column].values
        
        # Create sequences
        for i in range(len(group) - time_steps):
            sequences.append(features[i:i + time_steps])
            targets.append(target[i + time_steps])  # Target is the next lane change
    
    return np.array(sequences), np.array(targets)

def normalize_data(train_data, test_data=None, scaler=None, save_path=None):
    """
    Normalize the data using StandardScaler.
    
    Args:
        train_data (np.array): Training data to fit the scaler
        test_data (np.array, optional): Test data to transform
        scaler (StandardScaler, optional): Pre-trained scaler to use
        save_path (str, optional): Path to save the fitted scaler
        
    Returns:
        train_normalized (np.array): Normalized training data
        test_normalized (np.array, optional): Normalized test data if provided
        scaler (StandardScaler): Fitted scaler
    """
    # Reshape data if it's in sequence format [samples, time_steps, features]
    original_shape = train_data.shape
    if len(original_shape) == 3:
        train_data_reshaped = train_data.reshape(-1, original_shape[2])
        test_data_reshaped = test_data.reshape(-1, original_shape[2]) if test_data is not None else None
    else:
        train_data_reshaped = train_data
        test_data_reshaped = test_data
    
    # Fit or use provided scaler
    if scaler is None:
        scaler = StandardScaler()
        scaler.fit(train_data_reshaped)
    
    # Transform the data
    train_normalized_reshaped = scaler.transform(train_data_reshaped)
    
    # Reshape back to original shape
    if len(original_shape) == 3:
        train_normalized = train_normalized_reshaped.reshape(original_shape)
    else:
        train_normalized = train_normalized_reshaped
    
    # Save the scaler if requested
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump(scaler, save_path)
    
    # Transform test data if provided
    test_normalized = None
    if test_data is not None:
        test_normalized_reshaped = scaler.transform(test_data_reshaped)
        
        if len(original_shape) == 3:
            test_shape = test_data.shape
            test_normalized = test_normalized_reshaped.reshape(test_shape)
        else:
            test_normalized = test_normalized_reshaped
    
    return train_normalized, test_normalized, scaler