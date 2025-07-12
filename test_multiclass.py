import torch
from torch.utils.data import DataLoader
import os
import json
import numpy as np

# Import modules
from dataset import MultiClassEEGDataset
from dataset import EEGNetDataset
from model import create_eegnet_model
from utils import set_all_seeds
from training import evaluate_multiclass_model
from evaluation import plot_multiclass_results, analyze_misclassifications, plot_multiclass_predictions_distribution


def test_multiclass():
    """
    Test script for multi-class EEG classification model
    """
    # Set seeds for reproducibility
    set_all_seeds(42)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data directory
    data_dir = "C:\Github\OpenCloseFeet\Muse_data_OpenCloseFeet_segmented"
    
    print("\n" + "="*70)
    print("TESTING MULTI-CLASS EEG CLASSIFIER")
    print("="*70)
    
    # Load configuration
    config_path = 'models/multiclass_config.json'
    if not os.path.exists(config_path):
        print(f"ERROR: Configuration file not found at {config_path}")
        print("Please run main_multiclass.py first to train the model.")
        return
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print("\nLoaded configuration:")
    print(f"  Classes: {config['classes']}")
    print(f"  Normalization: {config['use_normalization']}")
    print(f"  Training accuracy: {config.get('best_val_accuracy', 'N/A'):.2f}%")
    
    # Load normalization parameters if used
    norm_params = None
    if config['use_normalization']:
        norm_params_path = config['normalization_params_file']
        if os.path.exists(norm_params_path):
            norm_params = MultiClassEEGDataset.load_normalization_params(norm_params_path)
            print(f"\nLoaded normalization parameters from {norm_params_path}")
        else:
            print(f"\nWARNING: Normalization parameters not found!")
    
    # Load test dataset
    print("\nLoading test data...")
    test_dataset = MultiClassEEGDataset(
        data_dir=data_dir,
        run_number=[1],  # Test run
        window_size=1025,
        debug=True,
        normalize=config['use_normalization'],
        normalization_params=norm_params
    )
    
    # Wrap for EEGNet
    eegnet_test = EEGNetDataset(test_dataset)
    test_loader = DataLoader(eegnet_test, batch_size=32, shuffle=False)
    
    # Load model
    model_path = 'best_multiclass_multiclass_model.pth'
    if not os.path.exists(model_path):
        print(f"\nERROR: Model file not found at {model_path}")
        print("Please run main_multiclass.py first to train the model.")
        return
    
    # Create model
    print("\nLoading multi-class model...")
    model = create_eegnet_model(
        task_type='multiclass',
        num_classes=3,
        samples=1025
    ).to(device)
    
    # Load weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Evaluate
    print("\nEvaluating model on test set...")
    test_results = evaluate_multiclass_model(model, test_loader, device)
    
    # Compare with training performance
    print("\n" + "="*50)
    print("PERFORMANCE COMPARISON")
    print("="*50)
    if 'test_accuracy' in config:
        print(f"Training test accuracy: {config['test_accuracy']:.2f}%")
        print(f"Current test accuracy: {test_results['accuracy']:.2f}%")
        diff = abs(config['test_accuracy'] - test_results['accuracy'])
        
        if diff < 2.0:
            print(f"✅ Performance is consistent (difference: {diff:.2f}%)")
        else:
            print(f"⚠️  Performance difference detected: {diff:.2f}%")
    
    if 'test_macro_auc' in config:
        print(f"\nTraining test macro AUC: {config['test_macro_auc']:.3f}")
        print(f"Current test macro AUC: {test_results['macro_auc']:.3f}")
    
    # Additional analysis
    print("\n" + "="*50)
    print("DETAILED ANALYSIS")
    print("="*50)
    
    # Misclassification patterns
    analyze_misclassifications(test_results)
    
    # Generate prediction distribution plot
    if not os.path.exists('results'):
        os.makedirs('results')
    
    plot_multiclass_predictions_distribution(
        test_results,
        save_path='results/test_multiclass_pred_dist.png'
    )
    
    # Class-wise performance
    print("\n" + "="*50)
    print("CLASS-WISE PERFORMANCE SUMMARY")
    print("="*50)
    
    cm = test_results['confusion_matrix']
    for i, class_name in enumerate(['Resting', 'Fists', 'Feet']):
        if cm[i].sum() > 0:
            accuracy = cm[i, i] / cm[i].sum() * 100
            print(f"{class_name}: {accuracy:.1f}% accuracy "
                  f"({cm[i, i]}/{cm[i].sum()} correct)")
    
    print("\n" + "="*70)
    print("TESTING COMPLETE")
    print("="*70)
    
    return test_results


def quick_inference_demo(n_samples=5):
    """
    Quick demo showing model predictions on a few samples
    """
    set_all_seeds(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = create_eegnet_model(task_type='multiclass', num_classes=3, samples=1025).to(device)
    model.load_state_dict(torch.load('best_multiclass_multiclass_model.pth', map_location=device))
    model.eval()
    
    # Load a few test samples
    data_dir = "C:\Github\OpenCloseFeet\Muse_data_OpenCloseFeet_segmented"
    test_dataset = MultiClassEEGDataset(
        data_dir=data_dir,
        run_number=[1],
        window_size=1025,
        debug=False,
        normalize=False  # Adjust based on training
    )
    
    eegnet_test = EEGNetDataset(test_dataset)
    test_loader = DataLoader(eegnet_test, batch_size=n_samples, shuffle=True)
    
    # Get one batch
    data, labels = next(iter(test_loader))
    data = data.to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(data)
        probs = torch.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, 1)
    
    # Display results
    class_names = ['Resting', 'Fists', 'Feet']
    print("\n" + "="*50)
    print("SAMPLE PREDICTIONS")
    print("="*50)
    
    for i in range(len(labels)):
        true_label = class_names[labels[i]]
        pred_label = class_names[preds[i]]
        confidence = probs[i, preds[i]].item()
        
        print(f"\nSample {i+1}:")
        print(f"  True: {true_label}")
        print(f"  Predicted: {pred_label} (confidence: {confidence:.3f})")
        print(f"  All probabilities: ", end="")
        for j, cls in enumerate(class_names):
            print(f"{cls}: {probs[i, j]:.3f}  ", end="")
        print()
    
    print("="*50)


if __name__ == "__main__":
    # Run full test
    results = test_multiclass()
    
    # Run quick demo
    print("\n\n")
    quick_inference_demo(n_samples=5)