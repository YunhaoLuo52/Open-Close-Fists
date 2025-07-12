import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, classification_report
import random
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import json

from dataset import TwoChannelEEGDataset, EEGNetDataset
from model import TwoChannelLSTMClassifier, create_eegnet_model
from utils import set_all_seeds, create_validation_split
from training import train_simple_model, train_simple_model_with_universal_best, compare_all_runs, train_eegnet_model
from evaluation import evaluate_simple_model, plot_results, print_classification_results, plot_results_short


def load_model_config(experiment_name):
    """Load model configuration"""
    config_path = f'models/{experiment_name}_model_config.json'
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    else:
        print(f"Warning: Config file {config_path} not found. Using defaults.")
        return None


def main(exp_number=1):
    # Set seeds for reproducibility
    set_all_seeds(42)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Configuration
    data_dir = "C:\Github\OpenCloseFeet\Muse_data_OpenCloseFeet_segmented"

    if exp_number == 1:
        task = "openclosefists"
        experiment_name = "fists"
    else:
        task = "openclosefeet"
        experiment_name = "feet"
    
    print(f"\n{'='*50}")
    print(f"Testing {experiment_name} classifier")
    print(f"{'='*50}")
    
    # Load model configuration
    config = load_model_config(experiment_name)
    
    # Determine if normalization was used during training
    use_normalization = False
    norm_params = None
    
    if config:
        use_normalization = config.get('use_normalization', False)
        print(f"\nModel was trained with normalization: {use_normalization}")
    
    # Load normalization parameters if needed
    if use_normalization:
        norm_params_path = f"normalization_params/{experiment_name}_norm_params.json"
        
        if os.path.exists(norm_params_path):
            norm_params = TwoChannelEEGDataset.load_normalization_params(norm_params_path)
            print(f"\nLoaded normalization parameters from {norm_params_path}")
            print(f"  TP9 - mean: {norm_params['tp9_mean']:.6f}, std: {norm_params['tp9_std']:.6f}")
            print(f"  TP10 - mean: {norm_params['tp10_mean']:.6f}, std: {norm_params['tp10_std']:.6f}")
        else:
            print(f"\nWARNING: Model expects normalization but parameters not found!")
            print(f"Expected at: {norm_params_path}")
            print("This will cause poor test performance!")
    else:
        print("\nModel was trained WITHOUT normalization - using raw data for testing")
    
    # Load test dataset with same preprocessing as training
    test_dataset = TwoChannelEEGDataset(
        data_dir=data_dir,
        exp_number=exp_number,
        run_number=[1],  # Test run
        task=task,
        window_size=1025,
        debug=True,
        normalize=use_normalization,  # Match training
        normalization_params=norm_params  # Use training params if normalized
    )

    # Wrap for EEGNet
    eegnet_test_dataset = EEGNetDataset(test_dataset)

    print(f"\nTest dataset: {len(test_dataset)} samples")

    # Create data loader
    batch_size = 32
    test_loader = DataLoader(eegnet_test_dataset, batch_size=batch_size, shuffle=False)

    # Load model
    model_path = f"best_{experiment_name}_eegnet_model.pth"
    
    if not os.path.exists(model_path):
        print(f"\nERROR: Model file {model_path} not found!")
        print("Please run main.py first to train the model.")
        return
    
    # Create model with same architecture
    model = create_eegnet_model(task_type='binary', num_classes=1, samples=1025).to(device)
    
    # Load trained weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"\nLoaded model from {model_path}")
    
    # Put model in evaluation mode
    model.eval()
    
    # Evaluate
    print(f"\nEvaluating model...")
    labels, preds, probs = evaluate_simple_model(model, test_loader, device)
    
    # Print classification report
    print("\nClassification Report:")
    print("=" * 50)
    print(classification_report(labels, preds, target_names=["Resting", f"Open/Close {experiment_name}"]))
    
    # Plot results and get AUC score
    auc_score = plot_results_short(labels, preds, probs, save_path=f'{experiment_name}_test_results.png')
    print(f"\nAUC Score: {auc_score:.3f}")
    
    # Additional statistics
    print(f"\nTest Dataset Statistics:")
    print(f"Total samples: {len(labels)}")
    print(f"Resting samples: {np.sum(labels == 0)} ({np.mean(labels == 0)*100:.1f}%)")
    print(f"Open/Close {experiment_name} samples: {np.sum(labels == 1)} ({np.mean(labels == 1)*100:.1f}%)")
    
    print(f"\nPrediction Statistics:")
    print(f"Predicted Resting: {np.sum(preds == 0)} ({np.mean(preds == 0)*100:.1f}%)")
    print(f"Predicted Open/Close {experiment_name}: {np.sum(preds == 1)} ({np.mean(preds == 1)*100:.1f}%)")
    
    # Compare with training performance if config available
    if config:
        print(f"\nPerformance Comparison:")
        print(f"  Training test AUC: {config.get('test_auc', 'N/A')}")
        print(f"  Current test AUC: {auc_score:.3f}")
        
        if 'test_auc' in config:
            diff = abs(config['test_auc'] - auc_score)
            if diff < 0.01:
                print(f"\n Performance matches training! (difference: {diff:.3f})")
            elif diff < 0.05:
                print(f"\n Performance is close to training (difference: {diff:.3f})")
            else:
                print(f"\n WARNING: Performance differs from training (difference: {diff:.3f})")
                print("  This might indicate a preprocessing mismatch!")
    
    print(f"\n{'='*50}")
    print(f"Testing complete for {experiment_name}")
    print(f"{'='*50}\n")
    
    return auc_score


if __name__ == "__main__":
    fists_auc = main(exp_number=1)
    feet_auc = main(exp_number=2)
    
    if fists_auc and feet_auc:
        print(f"\n{'='*50}")
        print(f"TEST SUMMARY")
        print(f"{'='*50}")
        print(f"Fists AUC: {fists_auc:.3f}")
        print(f"Feet AUC: {feet_auc:.3f}")
        print(f"{'='*50}")