import torch
from torch.utils.data import DataLoader, Subset
import os
import numpy as np

# Import custom modules
from dataset import MultiClassEEGDataset
from dataset import EEGNetDataset
from model import create_eegnet_model
from utils import set_all_seeds
from training import train_multiclass_eegnet, evaluate_multiclass_model, save_multiclass_config
from evaluation import plot_multiclass_results, plot_multiclass_predictions_distribution, analyze_misclassifications, create_multiclass_summary_report


def main_multiclass(use_normalization=False):
    """
    Main script for multi-class EEG classification (Rest vs Fists vs Feet)
    """
    # Set random seeds for reproducibility
    set_all_seeds(42)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data directory
    data_dir = "C:\Github\OpenCloseFeet\Muse_data_OpenCloseFeet_segmented"
    
    # Create directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    if use_normalization:
        os.makedirs("normalization_params", exist_ok=True)
    
    print("\n" + "="*70)
    print("MULTI-CLASS EEG CLASSIFICATION: Rest vs Fists vs Feet")
    print("="*70)
    
    # Load training dataset
    print("\nLoading training data...")
    train_dataset = MultiClassEEGDataset(
        data_dir=data_dir,
        run_number=[2],  # Training runs
        window_size=1025,
        debug=True,
        normalize=use_normalization
    )
    
    # Save normalization parameters if used
    norm_params = None
    if use_normalization:
        norm_params_path = "normalization_params/multiclass_norm_params.json"
        train_dataset.save_normalization_params(norm_params_path)
        norm_params = train_dataset.get_normalization_params()
    
    # Load test dataset with same preprocessing
    print("\nLoading test data...")
    test_dataset = MultiClassEEGDataset(
        data_dir=data_dir,
        run_number=[1],  # Test runs
        window_size=1025,
        debug=True,
        normalize=use_normalization,
        normalization_params=norm_params
    )
    
    # Create train/validation split
    train_indices, val_indices = train_dataset.create_stratified_split(test_size=0.15)
    
    print(f"\nDataset splits:")
    print(f"  Train: {len(train_indices)} samples")
    print(f"  Validation: {len(val_indices)} samples")
    print(f"  Test: {len(test_dataset)} samples")
    
    # Create subsets
    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(train_dataset, val_indices)
    
    # Wrap for EEGNet
    eegnet_train = EEGNetDataset(train_subset)
    eegnet_val = EEGNetDataset(val_subset)
    eegnet_test = EEGNetDataset(test_dataset)
    
    # Create data loaders
    batch_size = 32
    train_loader = DataLoader(eegnet_train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(eegnet_val, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(eegnet_test, batch_size=batch_size, shuffle=False)
    
    # Get class weights for balanced training
    class_weights = train_dataset.get_class_weights()
    print(f"\nClass weights for balanced training: {class_weights.numpy()}")
    
    # Create model - IMPORTANT: num_classes=3 for multi-class
    print("\nCreating multi-class EEGNet model...")
    model = create_eegnet_model(
        task_type='multiclass',  # Use multiclass variant
        num_classes=3,           # 3 classes: Rest, Fists, Feet
        samples=1025
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Train model
    print("\nStarting training...")
    train_losses, val_losses, val_accuracies, best_metrics = train_multiclass_eegnet(
        model, train_loader, val_loader, device,
        class_weights=class_weights,
        epochs=500,  # May need more epochs for multi-class
        lr=0.001,
        experiment_name='multiclass'
    )
    
    # Load best model for evaluation
    best_model_path = 'best_multiclass_multiclass_model.pth'
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    print(f"\nLoaded best model from {best_model_path}")
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_results = evaluate_multiclass_model(model, test_loader, device)
    
    # Save configuration
    config = {
        'task': 'multiclass',
        'classes': ['Resting', 'Fists', 'Feet'],
        'num_classes': 3,
        'window_size': 1025,
        'use_normalization': use_normalization,
        'normalization_params_file': 'normalization_params/multiclass_norm_params.json' if use_normalization else None,
        'model_architecture': 'EEGNet_MultiClass',
        'training_epochs': len(train_losses),
        'best_val_accuracy': max(val_accuracies),
        'test_accuracy': test_results['accuracy'],
        'test_macro_auc': test_results['macro_auc'],
        'class_weights': class_weights.numpy().tolist()
    }
    save_multiclass_config('multiclass', config)
    
    # Visualizations
    print("\nGenerating visualizations...")
    
    # Main results plot
    plot_multiclass_results(
        train_losses, val_losses, val_accuracies, test_results,
        save_path='results/multiclass_results.png'
    )
    
    # Prediction distribution plot
    plot_multiclass_predictions_distribution(
        test_results,
        save_path='results/multiclass_pred_dist.png'
    )
    
    # Misclassification analysis
    analyze_misclassifications(test_results)
    
    # Create summary report
    create_multiclass_summary_report(
        train_losses, val_losses, val_accuracies, test_results,
        save_path='results/multiclass_report.txt'
    )
    
    print("\n" + "="*70)
    print("MULTI-CLASS TRAINING COMPLETE")
    print("="*70)
    print(f"Test Accuracy: {test_results['accuracy']:.2f}%")
    print(f"Macro-average AUC: {test_results['macro_auc']:.3f}")
    print("\nPer-class AUC:")
    for class_name, auc in test_results['auc_scores'].items():
        print(f"  {class_name}: {auc:.3f}")
    print("="*70)
    
    return test_results


if __name__ == "__main__":
    # Run multi-class classification without normalization (to match binary setup)
    results = main_multiclass(use_normalization=False)
    
    # Optionally, you can also try with normalization
    # results_norm = main_multiclass(use_normalization=True)