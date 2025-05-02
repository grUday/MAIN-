# model/evaluate.py
import torch
import numpy as np
import os
import argparse
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import label_binarize
import pandas as pd
import seaborn as sns
import time

from model.lstm_model import LaneChangeLSTM

def evaluate_model(model_path, data_dir, output_dir=None):
    """
    Evaluate a trained lane change prediction model.
    
    Args:
        model_path (str): Path to the trained model
        data_dir (str): Directory with processed data
        output_dir (str, optional): Directory to save evaluation results
    """
    if output_dir is None:
        output_dir = os.path.dirname(model_path)
    os.makedirs(output_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load test data
    print("Loading test data...")
    X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
    y_test = np.load(os.path.join(data_dir, 'y_test.npy'))
    
    # Load metadata and model info
    metadata = joblib.load(os.path.join(data_dir, 'metadata.pkl'))
    try:
        model_info = joblib.load(os.path.join(os.path.dirname(model_path), 'model_info.pkl'))
        print(f"Model information loaded. Trained for {model_info['epochs_completed']} epochs.")
    except:
        print("Model info not found. Using default values.")
        model_info = {
            'input_size': X_test.shape[2],
            'hidden_size': 64,
            'num_layers': 2,
            'dropout': 0.2,
            'num_classes': 3
        }
    
    # Initialize model
    model = LaneChangeLSTM(
        input_size=model_info['input_size'],
        hidden_size=model_info['hidden_size'],
        num_layers=model_info['num_layers'],
        dropout=model_info['dropout'],
        num_classes=model_info['num_classes']
    ).to(device)
    
    # Load model weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Convert to PyTorch tensors
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    
    # Get predictions
    print("Making predictions...")
    start_time = time.time()
    
    with torch.no_grad():
        batch_size = 128  # Process in batches to avoid OOM for large datasets
        all_probs = []
        
        for i in range(0, len(X_test), batch_size):
            batch_x = X_test_tensor[i:i+batch_size]
            outputs = model(batch_x)
            probs = torch.softmax(outputs, dim=1)
            all_probs.append(probs.cpu().numpy())
    
    all_probs = np.vstack(all_probs)
    pred_classes = np.argmax(all_probs, axis=1)
    
    # Convert back to -1, 0, 1 format
    pred_classes = pred_classes - 1
    
    inference_time = time.time() - start_time
    print(f"Inference completed in {inference_time:.2f} seconds")
    print(f"Average inference time per sample: {(inference_time / len(X_test)) * 1000:.2f} ms")
    
    # Calculate metrics
    class_labels = {-1: 'Left', 0: 'None', 1: 'Right'}
    
    # Classification report
    report = classification_report(y_test, pred_classes, target_names=[class_labels[i] for i in [-1, 0, 1]])
    print("\nClassification Report:")
    print(report)
    
    # Save classification report
    with open(os.path.join(output_dir, 'eval_classification_report.txt'), 'w') as f:
        f.write(report)
        f.write(f"\nInference time: {inference_time:.2f} seconds")
        f.write(f"\nAverage inference time per sample: {(inference_time / len(X_test)) * 1000:.2f} ms")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, pred_classes)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=[class_labels[i] for i in [-1, 0, 1]],
                yticklabels=[class_labels[i] for i in [-1, 0, 1]])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'eval_confusion_matrix.png'))
    
    # ROC curves and AUC (one-vs-rest)
    y_test_bin = label_binarize(y_test + 1, classes=[0, 1, 2])
    n_classes = 3
    
    plt.figure(figsize=(10, 8))
    
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], all_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, 
                 label=f'ROC curve for {class_labels[i-1]} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_dir, 'eval_roc_curves.png'))
    
    # Precision-Recall curves
    plt.figure(figsize=(10, 8))
    
    for i in range(n_classes):
        precision, recall, _ = precision_recall_curve(y_test_bin[:, i], all_probs[:, i])
        plt.plot(recall, precision, lw=2,
                 label=f'Precision-Recall curve for {class_labels[i-1]}')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend(loc="lower left")
    plt.savefig(os.path.join(output_dir, 'eval_precision_recall_curves.png'))
    
    # Feature importance analysis (if possible)
    try:
        # This is a simplified approach - for LSTMs, feature importance analysis is more complex
        # Here we're just measuring input perturbation sensitivity
        print("Analyzing feature importance...")
        feature_columns = []
        with open(os.path.join(data_dir, 'feature_columns.txt'), 'r') as f:
            feature_columns = [line.strip() for line in f.readlines()]
        
        feature_importance = []
        baseline_preds = all_probs.copy()
        
        for i in range(X_test.shape[2]):
            perturbed_X = X_test.copy()
            # Shuffle the values in this feature
            perturbed_X[:, :, i] = np.random.permutation(perturbed_X[:, :, i])
            
            perturbed_tensor = torch.FloatTensor(perturbed_X).to(device)
            with torch.no_grad():
                perturbed_outputs = []
                for j in range(0, len(perturbed_tensor), batch_size):
                    batch_x = perturbed_tensor[j:j+batch_size]
                    outputs = model(batch_x)
                    probs = torch.softmax(outputs, dim=1)
                    perturbed_outputs.append(probs.cpu().numpy())
                
                perturbed_probs = np.vstack(perturbed_outputs)
            
            # Measure difference in predictions
            importance = np.mean(np.abs(baseline_preds - perturbed_probs))
            feature_importance.append(importance)
        
        # Plot feature importance
        if len(feature_columns) == len(feature_importance):
            plt.figure(figsize=(12, 8))
            indices = np.argsort(feature_importance)
            plt.barh(range(len(indices)), [feature_importance[i] for i in indices], align='center')
            plt.yticks(range(len(indices)), [feature_columns[i] for i in indices])
            plt.xlabel('Feature Importance')
            plt.title('Feature Importance Analysis')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'feature_importance.png'))
            
            # Save feature importance data
            importance_df = pd.DataFrame({
                'Feature': feature_columns,
                'Importance': feature_importance
            })
            importance_df = importance_df.sort_values('Importance', ascending=False)
            importance_df.to_csv(os.path.join(output_dir, 'feature_importance.csv'), index=False)
            
            print("Top 10 most important features:")
            print(importance_df.head(10))
    except Exception as e:
        print(f"Could not perform feature importance analysis: {e}")
    
    # Save evaluation results
    results = {
        'accuracy': (pred_classes == y_test).mean(),
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
        'inference_time': inference_time,
        'inference_time_per_sample': (inference_time / len(X_test)) * 1000
    }
    
    joblib.dump(results, os.path.join(output_dir, 'evaluation_results.pkl'))
    
    print(f"Evaluation completed. Results saved to {output_dir}")
    return results

def main():
    parser = argparse.ArgumentParser(description='Evaluate lane change prediction model')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to the trained model')
    parser.add_argument('--data_dir', type=str, default='../data/processed',
                        help='Directory with processed data')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save evaluation results')
    
    args = parser.parse_args()
    
    evaluate_model(args.model, args.data_dir, args.output_dir)

if __name__ == "__main__":
    main()