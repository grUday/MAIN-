# main.py
import os
import argparse
import time
import sys

def print_section(title):
    """Print a section header."""
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "="))
    print("=" * 80 + "\n")

def main():
    parser = argparse.ArgumentParser(description='Lane Change Prediction with CARLA and LSTM')
    parser.add_argument('--mode', type=str, default='all', 
                        choices=['all', 'collect', 'process', 'train', 'predict'],
                        help='Mode to run (all, collect, process, train, or predict)')
    parser.add_argument('--num_vehicles', type=int, default=20,
                        help='Number of vehicles to spawn in CARLA')
    parser.add_argument('--frames', type=int, default=10000,
                        help='Number of frames to record')
    parser.add_argument('--seq_length', type=int, default=30,
                        help='Sequence length for LSTM input')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--hidden_size', type=int, default=64,
                        help='Hidden size of LSTM layers')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='Number of LSTM layers')
    parser.add_argument('--load_model', type=str, default=None,
                        help='Path to pre-trained model to load for prediction')
    
    args = parser.parse_args()
    
    # Create directories if they don't exist
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('models/saved_models', exist_ok=True)
    
    start_time = time.time()
    
    if args.mode in ['all', 'collect']:
        print_section("DATA COLLECTION")
        # Update config with command line arguments
        from data_collection.config import DataCollectionConfig
        DataCollectionConfig.NUM_VEHICLES = args.num_vehicles
        DataCollectionConfig.FRAMES_TO_RECORD = args.frames
        
        # Run data collection
        from data_collection.collect_data import main as collect_main
        collect_main()
    
    if args.mode in ['all', 'process']:
        print_section("DATA PROCESSING")
        # Run data processing
        from data_processing.process_data import process_data
        process_data(
            input_file='data/raw/traffic_data.csv',
            output_dir='data/processed',
            sequence_length=args.seq_length
        )
    
    if args.mode in ['all', 'train']:
        print_section("MODEL TRAINING")
        # Run model training
        from model.train import train_model
        train_model(
            data_dir='data/processed',
            model_dir='models/saved_models',
            batch_size=args.batch_size,
            epochs=args.epochs,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers
        )
    
    if args.mode in ['all', 'predict']:
        print_section("PREDICTION AND VISUALIZATION")
        # Run prediction
        from prediction.predict import run_prediction
        model_path = args.load_model if args.load_model else 'models/saved_models/best_model.pth'
        run_prediction(
            model_path=model_path,
            data_dir='data/processed',
            num_vehicles=args.num_vehicles
        )
    
    total_time = time.time() - start_time
    print(f"\nTotal execution time: {total_time:.2f} seconds")

if __name__ == "__main__":
    main()