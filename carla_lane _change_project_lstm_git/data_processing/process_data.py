# data_processing/process_data.py
import pandas as pd
import numpy as np
import os
import argparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import joblib

from data_processing.utils import create_sequences, normalize_data

def process_data(input_file, output_dir, sequence_length=30, test_size=0.2, random_state=42):
    """
    Process raw CARLA data and prepare it for LSTM model training.
    
    Args:
        input_file (str): Path to input CSV file with raw CARLA data
        output_dir (str): Directory to save processed data
        sequence_length (int): Number of time steps in each input sequence
        test_size (float): Proportion of data to use for testing
        random_state (int): Random seed for reproducibility
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load raw data
    print(f"Loading data from {input_file}...")
    data = pd.read_csv(input_file)
    
    # Basic data cleaning
    print("Cleaning data...")
    # Drop rows with NaN values
    data = data.dropna()
    
    # Identify potential categorical columns
    print("Identifying and encoding categorical features...")
    categorical_columns = []
    encoders = {}
    
    for col in data.columns:
        if col not in ['timestamp', 'vehicle_id', 'lane_change']:
            # Check if column contains non-numeric data
            if data[col].dtype == 'object' or pd.api.types.is_categorical_dtype(data[col]):
                categorical_columns.append(col)
                # Use label encoding for categorical features
                encoder = LabelEncoder()
                data[col] = encoder.fit_transform(data[col])
                encoders[col] = encoder
    
    # Save encoders for later use
    if encoders:
        joblib.dump(encoders, os.path.join(output_dir, 'encoders.pkl'))
        
    # Create feature columns list
    feature_columns = [col for col in data.columns if col not in ['timestamp', 'vehicle_id', 'lane_change']]
    
    # Save feature columns for later use
    with open(os.path.join(output_dir, 'feature_columns.txt'), 'w') as f:
        for col in feature_columns:
            f.write(f"{col}\n")
    
    # Also save categorical columns info
    with open(os.path.join(output_dir, 'categorical_columns.txt'), 'w') as f:
        for col in categorical_columns:
            f.write(f"{col}\n")
    
    # Create sequences for time series prediction
    print(f"Creating sequences with length {sequence_length}...")
    X, y = create_sequences(data, sequence_length, target_column='lane_change')
    
    # Split data into training and testing sets
    print(f"Splitting data with test_size={test_size}...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Normalize the data
    print("Normalizing data...")
    X_train_norm, X_test_norm, scaler = normalize_data(
        X_train, X_test, save_path=os.path.join(output_dir, 'scaler.pkl')
    )
    
    # Save the processed data
    print("Saving processed data...")
    np.save(os.path.join(output_dir, 'X_train.npy'), X_train_norm)
    np.save(os.path.join(output_dir, 'X_test.npy'), X_test_norm)
    np.save(os.path.join(output_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(output_dir, 'y_test.npy'), y_test)
    
    # Save metadata
    metadata = {
        'sequence_length': sequence_length,
        'n_features': X_train.shape[2],
        'n_classes': len(np.unique(y)),
        'class_distribution': {int(c): int((y == c).sum()) for c in np.unique(y)},
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'categorical_columns': categorical_columns
    }
    
    # Save as text for readability
    with open(os.path.join(output_dir, 'metadata.txt'), 'w') as f:
        for key, value in metadata.items():
            f.write(f"{key}: {value}\n")
    
    print(f"Data processing complete. Files saved to {output_dir}")
    print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")
    
    # Also save metadata as joblib for easier loading
    joblib.dump(metadata, os.path.join(output_dir, 'metadata.pkl'))
    
    return metadata

def main():
    parser = argparse.ArgumentParser(description='Process CARLA data for lane change prediction')
    parser.add_argument('--input', type=str, default='../data/raw/traffic_data.csv', 
                        help='Path to input CSV file')
    parser.add_argument('--output', type=str, default='../data/processed/', 
                        help='Directory to save processed data')
    parser.add_argument('--seq_length', type=int, default=30, 
                        help='Sequence length for LSTM input')
    parser.add_argument('--test_size', type=float, default=0.2, 
                        help='Proportion of data to use for testing')
    parser.add_argument('--random_state', type=int, default=42, 
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    process_data(
        args.input, 
        args.output, 
        sequence_length=args.seq_length,
        test_size=args.test_size,
        random_state=args.random_state
    )

if __name__ == "__main__":
    main()