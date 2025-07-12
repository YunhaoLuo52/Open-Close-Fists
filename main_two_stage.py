import torch
from torch.utils.data import DataLoader, Subset
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import modules
from dataset import TwoStageEEGDataset
from dataset import EEGNetDataset
from model import create_eegnet_model
from utils import set_all_seeds
from training import (train_stage_model, evaluate_stage_model, 
                               evaluate_two_stage_system, save_two_stage_config)


def plot_two_stage_results(stage1_results, stage2_results, two_stage_results, 
                          save_path='two_stage_results.png'):
    """Plot comprehensive results for two-stage classification"""
    fig = plt.figure(figsize=(20, 10))
    
    # Stage 1 Confusion Matrix
    ax1 = plt.subplot(2, 4, 1)
    cm1 = stage1_results['confusion_matrix']
    sns.heatmap(cm1, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Rest', 'Motor'],
                yticklabels=['Rest', 'Motor'])
    ax1.set_title(f'Stage 1: Rest vs Motor\nAcc: {stage1_results["accuracy"]:.1f}%, AUC: {stage1_results["auc"]:.3f}')
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('True')
    
    # Stage 2 Confusion Matrix
    ax2 = plt.subplot(2, 4, 2)
    cm2 = stage2_results['confusion_matrix']
    sns.heatmap(cm2, annot=True, fmt='d', cmap='Oranges',
                xticklabels=['Fists', 'Feet'],
                yticklabels=['Fists', 'Feet'])
    ax2.set_title(f'Stage 2: Fists vs Feet\nAcc: {stage2_results["accuracy"]:.1f}%, AUC: {stage2_results["auc"]:.3f}')
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('True')
    
    # Combined 3-class Confusion Matrix
    ax3 = plt.subplot(2, 4, 3)
    cm3 = two_stage_results['confusion_matrix']
    sns.heatmap(cm3, annot=True, fmt='d', cmap='Greens',
                xticklabels=['Rest', 'Fists', 'Feet'],
                yticklabels=['Rest', 'Fists', 'Feet'])
    ax3.set_title(f'Combined Two-Stage System\nAcc: {two_stage_results["accuracy"]:.1f}%')
    ax3.set_xlabel('Predicted')
    ax3.set_ylabel('True')
    
    # Normalized Combined Confusion Matrix
    ax4 = plt.subplot(2, 4, 4)
    cm3_norm = cm3.astype('float') / cm3.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm3_norm, annot=True, fmt='.2f', cmap='Greens',
                xticklabels=['Rest', 'Fists', 'Feet'],
                yticklabels=['Rest', 'Fists', 'Feet'])
    ax4.set_title('Normalized Confusion Matrix')
    ax4.set_xlabel('Predicted')
    ax4.set_ylabel('True')
    
    # Stage 1 Probability Distribution
    ax5 = plt.subplot(2, 4, 5)
    stage1_probs = two_stage_results['all_stage1_probs']
    true_labels = two_stage_results['all_true_labels']
    
    for label in [0, 1, 2]:
        mask = true_labels == label
        if np.sum(mask) > 0:
            label_name = ['Rest', 'Fists', 'Feet'][label]
            ax5.hist(stage1_probs[mask], bins=20, alpha=0.5, density=True, label=f'True: {label_name}')
    
    ax5.set_xlabel('Stage 1: P(Motor Imagery)')
    ax5.set_ylabel('Density')
    ax5.set_title('Stage 1 Probability Distribution')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Stage 2 Probability Distribution (only for motor imagery)
    ax6 = plt.subplot(2, 4, 6)
    stage2_probs = two_stage_results['all_stage2_probs']
    motor_mask = true_labels > 0
    
    for label in [1, 2]:
        mask = (true_labels == label) & motor_mask
        if np.sum(mask) > 0:
            label_name = ['Rest', 'Fists', 'Feet'][label]
            valid_probs = stage2_probs[mask]
            valid_probs = valid_probs[valid_probs > 0]  # Only non-zero probs
            if len(valid_probs) > 0:
                ax6.hist(valid_probs, bins=20, alpha=0.5, density=True, label=f'True: {label_name}')
    
    ax6.set_xlabel('Stage 2: P(Feet)')
    ax6.set_ylabel('Density')
    ax6.set_title('Stage 2 Probability Distribution\n(Motor Imagery Only)')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # Per-class Performance Comparison
    ax7 = plt.subplot(2, 4, 7)
    
    # Original 3-class results (from your report)
    original_acc = [61.7, 65.6, 52.0]  # From confusion matrix diagonal
    
    # Two-stage results
    two_stage_acc = []
    for i in range(3):
        if cm3[i].sum() > 0:
            two_stage_acc.append(cm3[i, i] / cm3[i].sum() * 100)
        else:
            two_stage_acc.append(0)
    
    x = np.arange(3)
    width = 0.35
    
    bars1 = ax7.bar(x - width/2, original_acc, width, label='Original 3-class', alpha=0.8)
    bars2 = ax7.bar(x + width/2, two_stage_acc, width, label='Two-stage', alpha=0.8)
    
    ax7.set_xlabel('Class')
    ax7.set_ylabel('Recall (%)')
    ax7.set_title('Per-Class Recall Comparison')
    ax7.set_xticks(x)
    ax7.set_xticklabels(['Rest', 'Fists', 'Feet'])
    ax7.legend()
    ax7.grid(True, alpha=0.3, axis='y')
    
    # Error Analysis
    ax8 = plt.subplot(2, 4, 8)
    
    # Calculate error types
    pred_labels = two_stage_results['all_pred_labels']
    
    # Error categories
    rest_as_motor = np.sum((true_labels == 0) & (pred_labels > 0))
    motor_as_rest = np.sum((true_labels > 0) & (pred_labels == 0))
    fists_as_feet = np.sum((true_labels == 1) & (pred_labels == 2))
    feet_as_fists = np.sum((true_labels == 2) & (pred_labels == 1))
    
    error_types = ['Rest→Motor', 'Motor→Rest', 'Fists→Feet', 'Feet→Fists']
    error_counts = [rest_as_motor, motor_as_rest, fists_as_feet, feet_as_fists]
    
    ax8.bar(error_types, error_counts, alpha=0.8)
    ax8.set_xlabel('Error Type')
    ax8.set_ylabel('Count')
    ax8.set_title('Error Analysis')
    ax8.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, (x, y) in enumerate(zip(error_types, error_counts)):
        ax8.text(i, y + 0.5, str(y), ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def main_two_stage(use_normalization=False):
    """
    Main script for two-stage EEG classification
    Stage 1: Rest vs Motor Imagery
    Stage 2: Fists vs Feet
    """
    # Set random seeds
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
    print("TWO-STAGE EEG CLASSIFICATION")
    print("Stage 1: Rest vs Motor Imagery")
    print("Stage 2: Fists vs Feet")
    print("="*70)
    
    # ============ STAGE 1: Rest vs Motor Imagery ============
    print("\n" + "="*50)
    print("STAGE 1: Rest vs Motor Imagery")
    print("="*50)
    
    # Load Stage 1 training data
    stage1_train_dataset = TwoStageEEGDataset(
        data_dir=data_dir,
        run_number=[2],
        stage='stage1',
        window_size=1025,
        debug=True,
        normalize=use_normalization
    )
    
    # Save normalization parameters if used
    norm_params = None
    if use_normalization:
        norm_params_path = "normalization_params/two_stage_norm_params.json"
        stage1_train_dataset.save_normalization_params(norm_params_path)
        norm_params = stage1_train_dataset.get_normalization_params()
    
    # Load Stage 1 test data
    stage1_test_dataset = TwoStageEEGDataset(
        data_dir=data_dir,
        run_number=[1],
        stage='stage1',
        window_size=1025,
        debug=True,
        normalize=use_normalization,
        normalization_params=norm_params
    )
    
    # Create train/validation split
    train_indices, val_indices = stage1_train_dataset.create_stratified_split(test_size=0.15)
    
    print(f"\nStage 1 Dataset splits:")
    print(f"  Train: {len(train_indices)} samples")
    print(f"  Validation: {len(val_indices)} samples")
    print(f"  Test: {len(stage1_test_dataset)} samples")
    
    # Create data loaders for Stage 1
    train_subset = Subset(stage1_train_dataset, train_indices)
    val_subset = Subset(stage1_train_dataset, val_indices)
    
    eegnet_train = EEGNetDataset(train_subset)
    eegnet_val = EEGNetDataset(val_subset)
    eegnet_test = EEGNetDataset(stage1_test_dataset)
    
    batch_size = 32
    stage1_train_loader = DataLoader(eegnet_train, batch_size=batch_size, shuffle=True)
    stage1_val_loader = DataLoader(eegnet_val, batch_size=batch_size, shuffle=False)
    stage1_test_loader = DataLoader(eegnet_test, batch_size=batch_size, shuffle=False)
    
    # Get class weights
    stage1_class_weights = stage1_train_dataset.get_class_weights()
    print(f"\nStage 1 class weights: {stage1_class_weights.numpy()}")
    
    # Create and train Stage 1 model
    stage1_model = create_eegnet_model(
        task_type='binary',
        num_classes=1,  # Binary classification
        samples=1025
    ).to(device)
    
    # Train Stage 1
    stage1_train_losses, stage1_val_losses, stage1_val_accs, stage1_best_metrics = train_stage_model(
        stage1_model, stage1_train_loader, stage1_val_loader, device,
        stage_name='stage1',
        class_weights=stage1_class_weights,
        epochs=300,
        lr=0.001
    )
    
    # Load best Stage 1 model and evaluate
    stage1_model.load_state_dict(torch.load('best_stage1_model.pth', map_location=device))
    stage1_results = evaluate_stage_model(
        stage1_model, stage1_test_loader, device, 
        'Stage 1', ['Rest', 'Motor Imagery']
    )
    
    # ============ STAGE 2: Fists vs Feet ============
    print("\n" + "="*50)
    print("STAGE 2: Fists vs Feet (Motor Imagery Only)")
    print("="*50)
    
    # Load Stage 2 training data
    stage2_train_dataset = TwoStageEEGDataset(
        data_dir=data_dir,
        run_number=[2],
        stage='stage2',
        window_size=1025,
        debug=True,
        normalize=use_normalization,
        normalization_params=norm_params
    )
    
    # Load Stage 2 test data
    stage2_test_dataset = TwoStageEEGDataset(
        data_dir=data_dir,
        run_number=[1],
        stage='stage2',
        window_size=1025,
        debug=True,
        normalize=use_normalization,
        normalization_params=norm_params
    )
    
    # Create train/validation split for Stage 2
    train_indices2, val_indices2 = stage2_train_dataset.create_stratified_split(test_size=0.15)
    
    print(f"\nStage 2 Dataset splits:")
    print(f"  Train: {len(train_indices2)} samples")
    print(f"  Validation: {len(val_indices2)} samples")
    print(f"  Test: {len(stage2_test_dataset)} samples")
    
    # Create data loaders for Stage 2
    train_subset2 = Subset(stage2_train_dataset, train_indices2)
    val_subset2 = Subset(stage2_train_dataset, val_indices2)
    
    eegnet_train2 = EEGNetDataset(train_subset2)
    eegnet_val2 = EEGNetDataset(val_subset2)
    eegnet_test2 = EEGNetDataset(stage2_test_dataset)
    
    stage2_train_loader = DataLoader(eegnet_train2, batch_size=batch_size, shuffle=True)
    stage2_val_loader = DataLoader(eegnet_val2, batch_size=batch_size, shuffle=False)
    stage2_test_loader = DataLoader(eegnet_test2, batch_size=batch_size, shuffle=False)
    
    # Get class weights
    stage2_class_weights = stage2_train_dataset.get_class_weights()
    print(f"\nStage 2 class weights: {stage2_class_weights.numpy()}")
    
    # Create and train Stage 2 model
    stage2_model = create_eegnet_model(
        task_type='binary',
        num_classes=1,  # Binary classification
        samples=1025
    ).to(device)
    
    # Train Stage 2
    stage2_train_losses, stage2_val_losses, stage2_val_accs, stage2_best_metrics = train_stage_model(
        stage2_model, stage2_train_loader, stage2_val_loader, device,
        stage_name='stage2',
        class_weights=stage2_class_weights,
        epochs=300,
        lr=0.001
    )
    
    # Load best Stage 2 model and evaluate
    stage2_model.load_state_dict(torch.load('best_stage2_model.pth', map_location=device))
    stage2_results = evaluate_stage_model(
        stage2_model, stage2_test_loader, device,
        'Stage 2', ['Fists', 'Feet']
    )
    
    # ============ COMBINED TWO-STAGE EVALUATION ============
    print("\n" + "="*50)
    print("COMBINED TWO-STAGE SYSTEM EVALUATION")
    print("="*50)
    
    # Evaluate on the complete test set
    two_stage_results = evaluate_two_stage_system(
        stage1_model, stage2_model, 
        stage1_test_loader, device, 
        stage1_test_dataset
    )
    
    # Save configuration
    config = {
        'architecture': 'two_stage',
        'stage1': {
            'task': 'rest_vs_motor',
            'accuracy': stage1_results['accuracy'],
            'auc': stage1_results['auc'],
            'best_epoch': stage1_best_metrics['epoch']
        },
        'stage2': {
            'task': 'fists_vs_feet',
            'accuracy': stage2_results['accuracy'],
            'auc': stage2_results['auc'],
            'best_epoch': stage2_best_metrics['epoch']
        },
        'combined': {
            'accuracy': two_stage_results['accuracy'],
            'stage1_accuracy': two_stage_results['stage1_accuracy'],
            'stage2_accuracy': two_stage_results['stage2_accuracy']
        },
        'use_normalization': use_normalization,
        'window_size': 1025
    }
    save_two_stage_config(config)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_two_stage_results(
        stage1_results, stage2_results, two_stage_results,
        save_path='results/two_stage_results.png'
    )
    
    # Create detailed report
    with open('results/two_stage_report.txt', 'w') as f:
        f.write("="*60 + "\n")
        f.write("TWO-STAGE EEG CLASSIFICATION REPORT\n")
        f.write("="*60 + "\n\n")
        
        f.write("STAGE 1: Rest vs Motor Imagery\n")
        f.write("-"*40 + "\n")
        f.write(f"Accuracy: {stage1_results['accuracy']:.2f}%\n")
        f.write(f"AUC: {stage1_results['auc']:.3f}\n\n")
        
        f.write("STAGE 2: Fists vs Feet\n")
        f.write("-"*40 + "\n")
        f.write(f"Accuracy: {stage2_results['accuracy']:.2f}%\n")
        f.write(f"AUC: {stage2_results['auc']:.3f}\n\n")
        
        f.write("COMBINED TWO-STAGE SYSTEM\n")
        f.write("-"*40 + "\n")
        f.write(f"Overall Accuracy: {two_stage_results['accuracy']:.2f}%\n")
        f.write(f"Stage 1 Component Accuracy: {two_stage_results['stage1_accuracy']:.2f}%\n")
        if two_stage_results['stage2_accuracy']:
            f.write(f"Stage 2 Component Accuracy: {two_stage_results['stage2_accuracy']:.2f}%\n")
        
        f.write("\nCLASSIFICATION REPORT\n")
        f.write("-"*40 + "\n")
        f.write(two_stage_results['classification_report'])
    
    print("\n" + "="*70)
    print("TWO-STAGE TRAINING COMPLETE")
    print("="*70)
    print(f"Stage 1 (Rest vs Motor): {stage1_results['accuracy']:.2f}% accuracy, {stage1_results['auc']:.3f} AUC")
    print(f"Stage 2 (Fists vs Feet): {stage2_results['accuracy']:.2f}% accuracy, {stage2_results['auc']:.3f} AUC")
    print(f"Combined System: {two_stage_results['accuracy']:.2f}% accuracy")
    print("="*70)
    
    # Compare with original 3-class
    print("\nIMPROVEMENT ANALYSIS:")
    print(f"Original 3-class accuracy: 59.51%")
    print(f"Two-stage accuracy: {two_stage_results['accuracy']:.2f}%")
    print(f"Improvement: {two_stage_results['accuracy'] - 59.51:.2f}%")
    
    return two_stage_results


if __name__ == "__main__":
    results = main_two_stage(use_normalization=False)