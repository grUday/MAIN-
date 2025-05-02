# model/train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import time
import joblib

from model.lstm_model import LaneChangeLSTM

def train_model(data_dir, model_dir, batch_size=64, epochs=30, learning_rate=0.001, hidden_size=64, num_layers=2, dropout=0.2, early_stop_patience=5):
    """
    Train the LSTM model for lane change prediction.
    
    Args:
        data_dir (str): Directory with processed data
        model_dir (str): Directory to save trained model
        batch_size (int): Batch size for training
        epochs (int): Number of training epochs
        learning_rate (float): Learning rate
        hidden_size (int): Hidden size of LSTM
        num_layers (int): Number of LSTM layers
        dropout (float): Dropout rate
        early_stop_patience (int): Patience for early stopping
    """
    # Create model directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load processed data
    print("Loading processed data...")
    X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
    X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
    y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
    y_test = np.load(os.path.join(data_dir, 'y_test.npy'))
    
    # Load metadata
    metadata = joblib.load(os.path.join(data_dir, 'metadata.pkl'))
    
    # Class mapping: Convert -1, 0, 1 to 0, 1, 2 for CrossEntropyLoss
    y_train = np.array([y + 1 for y in y_train])
    y_test = np.array([y + 1 for y in y_test])
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.LongTensor(y_test)
    
    # Create dataloaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Initialize model
    input_size = X_train.shape[2]
    num_classes = 3  # left, none, right
    
    model = LaneChangeLSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        num_classes=num_classes
    ).to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    
    # Training loop
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model_path = os.path.join(model_dir, 'best_model.pth')
    patience_counter = 0
    
    print(f"Starting training for {epochs} epochs...")
    start_time = time.time()
    
    for epoch in range(epochs):
        # Training
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total
        train_losses.append(epoch_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_epoch_loss = val_loss / len(test_loader.dataset)
        val_epoch_acc = val_correct / val_total
        val_losses.append(val_epoch_loss)
        
        # Update learning rate
        scheduler.step(val_epoch_loss)
        
        # Print statistics
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f} | "
              f"Val Loss: {val_epoch_loss:.4f} | Val Acc: {val_epoch_acc:.4f}")
        
        # Check if this is the best model
        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved new best model with validation loss: {best_val_loss:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Load best model for evaluation
    model.load_state_dict(torch.load(best_model_path))
    
    # Full evaluation on test set
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Convert back to -1, 0, 1 format for reporting
    all_preds = np.array(all_preds) - 1
    all_labels = np.array(all_labels) - 1
    
    # Generate classification report
    class_labels = {-1: 'Left', 0: 'None', 1: 'Right'}
    report = classification_report(all_labels, all_preds, target_names=[class_labels[i] for i in [-1, 0, 1]])
    print("\nClassification Report:")
    print(report)
    
    # Save classification report
    with open(os.path.join(model_dir, 'classification_report.txt'), 'w') as f:
        f.write(report)
    
    # Generate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(3)
    plt.xticks(tick_marks, [class_labels[i] for i in [-1, 0, 1]])
    plt.yticks(tick_marks, [class_labels[i] for i in [-1, 0, 1]])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, 'confusion_matrix.png'))
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Loss History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, 'training_history.png'))
    
    # Save model architecture and hyperparameters
    model_info = {
        'input_size': input_size,
        'hidden_size': hidden_size,
        'num_layers': num_layers,
        'dropout': dropout,
        'num_classes': num_classes,
        'best_val_loss': best_val_loss,
        'training_time': training_time,
        'epochs_completed': epoch + 1
    }
    
    joblib.dump(model_info, os.path.join(model_dir, 'model_info.pkl'))
    
    print(f"\nModel saved to {best_model_path}")
    print(f"Training metrics saved to {model_dir}")
    
    return model, model_info

def main():
    parser = argparse.ArgumentParser(description='Train lane change prediction model')
    parser.add_argument('--data_dir', type=str, default='../data/processed',
                        help='Directory with processed data')
    parser.add_argument('--model_dir', type=str, default='../models/saved_models',
                        help='Directory to save trained model')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--hidden_size', type=int, default=64,
                        help='Hidden size of LSTM layers')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='Number of LSTM layers')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout rate')
    parser.add_argument('--early_stop_patience', type=int, default=5,
                        help='Patience for early stopping')
    
    args = parser.parse_args()
    
    train_model(
        args.data_dir,
        args.model_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        early_stop_patience=args.early_stop_patience
    )

if __name__ == "__main__":
    main()