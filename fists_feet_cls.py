import torch
from torch.utils.data import DataLoader, Subset
import os
import json
import numpy as np

from dataset import EEGNetDataset
from utils import set_all_seeds
from model import create_eegnet_model
from training import train_eegnet_model
from evaluation import evaluate_simple_model, plot_results, print_classification_results

from dataset import MultiClassEEGDataset  # <-- add this to your dataset.py

def main(use_normalization=True):
    # Set random seed and device
    set_all_seeds(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_dir = "C:/Github/OpenCloseFeet/Muse_data_OpenCloseFeet_segmented"
    model_name = "multiclass"

    os.makedirs("models", exist_ok=True)
    if use_normalization:
        os.makedirs("normalization_params", exist_ok=True)

    print(f"\n{'='*50}")
    print(f"Training EEGNet multiclass classifier")
    print(f"Normalization: {'ENABLED' if use_normalization else 'DISABLED'}")
    print(f"{'='*50}")

    # Load training dataset (run 2), and compute normalization
    train_dataset_full = MultiClassEEGDataset(
        data_dir=data_dir,
        run_numbers=[2],
        window_size=1025,
        overlap=0.5,
        debug=True,
        normalize=use_normalization,
        normalization_params=None
    )

    # Save normalization params
    norm_params = train_dataset_full.get_normalization_params()
    if use_normalization and norm_params:
        norm_file = f"normalization_params/{model_name}_norm_params.json"
        with open(norm_file, 'w') as f:
            json.dump(norm_params, f, indent=2)

    # Train/val split
    train_indices, val_indices = train_dataset_full.create_proper_train_test_split(test_size=0.1)
    train_dataset = Subset(train_dataset_full, train_indices)
    val_dataset = Subset(train_dataset_full, val_indices)

    # Load test dataset (run 1), using training normalization
    test_dataset = MultiClassEEGDataset(
        data_dir=data_dir,
        run_numbers=[1],
        window_size=1025,
        overlap=0.5,
        debug=True,
        normalize=use_normalization,
        normalization_params=norm_params
    )

    print(f"\nDataset splits:")
    print(f"  Train: {len(train_dataset)}")
    print(f"  Val:   {len(val_dataset)}")
    print(f"  Test:  {len(test_dataset)}")

    # Wrap with EEGNet format
    train_loader = DataLoader(EEGNetDataset(train_dataset), batch_size=32, shuffle=True)
    val_loader = DataLoader(EEGNetDataset(val_dataset), batch_size=32, shuffle=False)
    test_loader = DataLoader(EEGNetDataset(test_dataset), batch_size=32, shuffle=False)

    # Create multiclass EEGNet model
    model = create_eegnet_model(task_type='multiclass', num_classes=3, samples=1025).to(device)

    # Train model
    print(f"\nTraining EEGNet model...")
    train_losses, val_losses, best_acc = train_eegnet_model(
        model, train_loader, val_loader, device,
        epochs=700, lr=0.001, experiment_name=model_name
    )

    best_model_path = f"best_{model_name}_eegnet_model.pth"
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    print(f"Loaded best model from {best_model_path}")

    # Evaluate on test set
    print(f"\nEvaluating on test set...")
    labels, preds, probs = evaluate_simple_model(model, test_loader, device)
    print_classification_results(labels, preds, probs, name=model_name)

    # Plot and save results
    auc_score = plot_results(train_losses, val_losses, labels, preds, probs, f"{model_name}_train_result.png")

    config = {
        'task_type': 'multiclass',
        'num_classes': 3,
        'samples': 1025,
        'use_normalization': use_normalization,
        'normalization_params_file': norm_file if use_normalization else None,
        'window_size': 1025,
        'best_val_acc': best_acc,
        'test_auc': auc_score
    }

    with open(f"models/{model_name}_model_config.json", 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\n{'='*50}")
    print(f"Training complete for {model_name}")
    print(f"  Best Val Acc: {best_acc:.4f}")
    print(f"  Test AUC: {auc_score:.4f}")
    print(f"  Model saved: {best_model_path}")
    print(f"  Config saved: models/{model_name}_model_config.json")
    print(f"{'='*50}\n")

    return auc_score


if __name__ == "__main__":
    main(use_normalization=True)
