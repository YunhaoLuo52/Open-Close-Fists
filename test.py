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

from dataset import TwoChannelEEGDataset
from model import TwoChannelLSTMClassifier
from utils import set_all_seeds, create_validation_split
from training import train_simple_model
from evaluation import evaluate_simple_model, plot_results, print_classification_results,plot_results_short


def main(exp_number=1):
    # Set seeds for reproducibility
    set_all_seeds(42)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Configuration
    data_dir = "C:\Github\OpenCloseFeet\Muse_data_OpenCloseFeet_segmented"  # Path to data directory
    #data_dir = "D:\Faculty\ColumbiaUniversity\dataprocess\EEG\dataProcessing\Interaxon\museS\LSL\Python\muse-lsl-python\Data\sub-EB-43\Muse_data_OpenCloseFistsFeet_session1_segmented"

    if exp_number == 1:
        task = "openclosefists"
        experiment_name = "fists"
    else:
        task = "openclosefeet"
        experiment_name = "feet"        

    # Fists cls
    test_dataset = TwoChannelEEGDataset(
        data_dir=data_dir,
        exp_number=exp_number,
        run_number=[1],  # load both runs
        task=task,
        window_size=1025,
        debug=False
    )

    model_path = f"C:/Github/Open-Close/models/{experiment_name}_universal_best.pth"  # Path to saved model weights
    # model_path = f"C:/Github/Open-Close/models/{experiment_name}_current_run.pth"  # Path to saved model weights

    model = TwoChannelLSTMClassifier(input_channels=2, num_classes=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    print(f"Final test: {len(test_dataset)} samples")

    # Data loaders
    batch_size = 32
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Load model and evaluate
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
        


if __name__ == "__main__":
    main(exp_number=1)
    main(exp_number=2)
    
