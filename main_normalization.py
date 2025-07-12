import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import os

# Import custom modules
from dataset import TwoChannelEEGDataset, EEGNetDataset
from model import TwoChannelLSTMClassifier, create_eegnet_model
from utils import set_all_seeds, create_validation_split
from training import train_simple_model, train_simple_model_with_universal_best, compare_all_runs, train_eegnet_model
from evaluation import evaluate_simple_model, plot_results, print_classification_results


def main(exp_number=1, use_normalization=False):
    """
    Main training script with optional normalization
    
    Args:
        exp_number: 1 for fists, 2 for feet
        use_normalization: Whether to normalize the data (default: False to match original)
    """
    # Set random seeds for reproducibility
    set_all_seeds(42)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Data directory - modify this path as needed
    data_dir = "C:\Github\OpenCloseFeet\Muse_data_OpenCloseFeet_segmented"

    if exp_number == 1:
        task = "openclosefists"
        experiment_name = "fists"
    else:
        task = "openclosefeet"
        experiment_name = "feet"
    
    # Create directories for saving models and parameters
    os.makedirs("models", exist_ok=True)
    if use_normalization:
        os.makedirs("normalization_params", exist_ok=True)
    
    print(f"\n{'='*50}")
    print(f"Training {experiment_name} classifier")
    print(f"Normalization: {'ENABLED' if use_normalization else 'DISABLED'}")
    print(f"{'='*50}")
    
    # Load training dataset
    dataset = TwoChannelEEGDataset(
        data_dir=data_dir,
        exp_number=exp_number,
        run_number=[2],  # Training runs
        task=task,
        window_size=1025,
        debug=True,
        normalize=use_normalization,
        normalization_params=None  # Will compute from training data if normalizing
    )
    
    # Handle normalization parameters if needed
    norm_params = None
    if use_normalization:
        # Save normalization parameters for testing
        norm_params_path = f"normalization_params/{experiment_name}_norm_params.json"
        dataset.save_normalization_params(norm_params_path)
        norm_params = dataset.get_normalization_params()
    
    # Load test dataset with same preprocessing
    test_dataset = TwoChannelEEGDataset(
        data_dir=data_dir,
        exp_number=exp_number,
        run_number=[1],  # Test runs
        task=task,
        window_size=1025,
        debug=True,
        normalize=use_normalization,
        normalization_params=norm_params  # Use training normalization params if any
    )

    # Create proper train/test split
    train_indices, val_indices = dataset.create_proper_train_test_split(method='trial_based', test_size=0.1)

    print(f"\nDataset splits:")
    print(f"  Train: {len(train_indices)} samples")
    print(f"  Validation: {len(val_indices)} samples")
    print(f"  Test: {len(test_dataset)} samples")

    # Create datasets using Subset
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    # Wrap datasets for EEGNet
    eegnet_train_dataset = EEGNetDataset(train_dataset)
    eegnet_val_dataset = EEGNetDataset(val_dataset)
    eegnet_test_dataset = EEGNetDataset(test_dataset)

    # Create data loaders
    batch_size = 32
    train_loader = DataLoader(eegnet_train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(eegnet_val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(eegnet_test_dataset, batch_size=batch_size, shuffle=False)

    # Create model
    model = create_eegnet_model(task_type='binary', num_classes=1, samples=1025).to(device)
    
    print(f"\nTraining EEGNet model...")
    train_losses, val_losses, best_acc = train_eegnet_model(
        model, train_loader, val_loader, device, 
        epochs=700, lr=0.001, experiment_name=experiment_name
    )

    # Load best model for evaluation
    best_model_path = f'best_{experiment_name}_eegnet_model.pth'
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    print(f"\nLoaded best model from {best_model_path}")

    # Evaluate model on test set
    print(f"\nEvaluating on test set...")
    labels, preds, probs = evaluate_simple_model(model, test_loader, device)

    # Print results
    print_classification_results(labels, preds, probs, name=experiment_name)

    # Plot and save results
    auc_score = plot_results(train_losses, val_losses, labels, preds, probs, f'{experiment_name}_train_result.png')
    
    # Save model configuration for reproducibility
    model_config = {
        'task_type': 'binary',
        'num_classes': 1,
        'samples': 1025,
        'use_normalization': use_normalization,
        'normalization_params_file': f"normalization_params/{experiment_name}_norm_params.json" if use_normalization else None,
        'window_size': 1025,
        'experiment_name': experiment_name,
        'task': task,
        'exp_number': exp_number,
        'best_val_acc': best_acc,
        'test_auc': auc_score
    }
    
    import json
    with open(f'models/{experiment_name}_model_config.json', 'w') as f:
        json.dump(model_config, f, indent=2)
    
    print(f"\n{'='*50}")
    print(f"Training complete for {experiment_name}")
    print(f"  Best validation accuracy: {best_acc:.4f}")
    print(f"  Test AUC: {auc_score:.4f}")
    print(f"  Model saved to: {best_model_path}")
    print(f"  Config saved to: models/{experiment_name}_model_config.json")
    print(f"{'='*50}\n")
    
    return auc_score


if __name__ == "__main__":
    # Train WITHOUT normalization (to match your original setup)
    print("\n" + "="*70)
    print("TRAINING WITH NORMALIZATION")
    print("="*70)
    
    fists_auc = main(exp_number=1, use_normalization=True)
    feet_auc = main(exp_number=2, use_normalization=True)
    
    print(f"\n{'='*50}")
    print(f"SUMMARY")
    print(f"{'='*50}")
    print(f"Fists AUC: {fists_auc:.4f}")
    print(f"Feet AUC: {feet_auc:.4f}")
    print(f"{'='*50}")