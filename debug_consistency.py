import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
import os
import json

from dataset import TwoChannelEEGDataset, EEGNetDataset
from model import create_eegnet_model
from utils import set_all_seeds


def debug_data_consistency(exp_number=1):
    """Debug script to verify data consistency between training and testing"""
    
    set_all_seeds(42)
    
    data_dir = "C:\Github\OpenCloseFeet\Muse_data_OpenCloseFeet_segmented"
    
    if exp_number == 1:
        task = "openclosefists"
        experiment_name = "fists"
    else:
        task = "openclosefeet"
        experiment_name = "feet"
    
    print(f"\n{'='*60}")
    print(f"DEBUGGING DATA CONSISTENCY FOR {experiment_name.upper()}")
    print(f"{'='*60}")
    
    # 1. Load raw data without normalization
    print("\n1. LOADING RAW DATA (NO NORMALIZATION)")
    print("-" * 40)
    
    raw_train_dataset = TwoChannelEEGDataset(
        data_dir=data_dir,
        exp_number=exp_number,
        run_number=[2],
        task=task,
        window_size=1025,
        debug=False,
        normalize=False
    )
    
    raw_test_dataset = TwoChannelEEGDataset(
        data_dir=data_dir,
        exp_number=exp_number,
        run_number=[1],
        task=task,
        window_size=1025,
        debug=False,
        normalize=False
    )
    
    # Get statistics
    train_data_raw = raw_train_dataset.segments
    test_data_raw = raw_test_dataset.segments
    
    print(f"Raw Train Data:")
    print(f"  Shape: {train_data_raw.shape}")
    print(f"  Range: [{np.min(train_data_raw):.6f}, {np.max(train_data_raw):.6f}]")
    print(f"  Mean: {np.mean(train_data_raw):.6f}")
    print(f"  Std: {np.std(train_data_raw):.6f}")
    
    print(f"\nRaw Test Data:")
    print(f"  Shape: {test_data_raw.shape}")
    print(f"  Range: [{np.min(test_data_raw):.6f}, {np.max(test_data_raw):.6f}]")
    print(f"  Mean: {np.mean(test_data_raw):.6f}")
    print(f"  Std: {np.std(test_data_raw):.6f}")
    
    # 2. Check if normalization parameters exist
    print("\n2. CHECKING NORMALIZATION PARAMETERS")
    print("-" * 40)
    
    norm_params_path = f"normalization_params/{experiment_name}_norm_params.json"
    
    if os.path.exists(norm_params_path):
        with open(norm_params_path, 'r') as f:
            norm_params = json.load(f)
        print(f"Found normalization parameters:")
        for key, value in norm_params.items():
            print(f"  {key}: {value:.6f}")
    else:
        print(f"WARNING: No normalization parameters found at {norm_params_path}")
        print("Cannot proceed with normalized data comparison!")
        return
    
    # 3. Load normalized data
    print("\n3. LOADING NORMALIZED DATA")
    print("-" * 40)
    
    norm_train_dataset = TwoChannelEEGDataset(
        data_dir=data_dir,
        exp_number=exp_number,
        run_number=[2],
        task=task,
        window_size=1025,
        debug=False,
        normalize=True,
        normalization_params=None  # Compute from data
    )
    
    norm_test_dataset = TwoChannelEEGDataset(
        data_dir=data_dir,
        exp_number=exp_number,
        run_number=[1],
        task=task,
        window_size=1025,
        debug=False,
        normalize=True,
        normalization_params=norm_params  # Use training params
    )
    
    train_data_norm = norm_train_dataset.segments
    test_data_norm = norm_test_dataset.segments
    
    print(f"Normalized Train Data:")
    print(f"  Shape: {train_data_norm.shape}")
    print(f"  Range: [{np.min(train_data_norm):.6f}, {np.max(train_data_norm):.6f}]")
    print(f"  Mean: {np.mean(train_data_norm):.6f}")
    print(f"  Std: {np.std(train_data_norm):.6f}")
    
    print(f"\nNormalized Test Data:")
    print(f"  Shape: {test_data_norm.shape}")
    print(f"  Range: [{np.min(test_data_norm):.6f}, {np.max(test_data_norm):.6f}]")
    print(f"  Mean: {np.mean(test_data_norm):.6f}")
    print(f"  Std: {np.std(test_data_norm):.6f}")
    
    # 4. Check model file
    print("\n4. CHECKING MODEL FILE")
    print("-" * 40)
    
    model_path = f"best_{experiment_name}_eegnet_model.pth"
    
    if os.path.exists(model_path):
        print(f"Found model file: {model_path}")
        
        # Load model and check
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = create_eegnet_model(task_type='binary', num_classes=1, samples=1025).to(device)
        
        # Try loading weights
        try:
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)
            print("Successfully loaded model weights")
            
            # Check a few layer dimensions
            print("\nModel architecture check:")
            for name, param in list(model.named_parameters())[:5]:
                print(f"  {name}: {param.shape}")
                
        except Exception as e:
            print(f"ERROR loading model: {e}")
    else:
        print(f"ERROR: Model file not found at {model_path}")
    
    # 5. Quick inference test
    print("\n5. QUICK INFERENCE TEST")
    print("-" * 40)
    
    if os.path.exists(model_path):
        # Test on a few samples
        model.eval()
        
        # Create mini dataloaders
        eegnet_train = EEGNetDataset(norm_train_dataset)
        eegnet_test = EEGNetDataset(norm_test_dataset)
        
        train_loader = DataLoader(eegnet_train, batch_size=4, shuffle=False)
        test_loader = DataLoader(eegnet_test, batch_size=4, shuffle=False)
        
        with torch.no_grad():
            # Get one batch from each
            train_batch, train_labels = next(iter(train_loader))
            test_batch, test_labels = next(iter(test_loader))
            
            train_batch = train_batch.to(device)
            test_batch = test_batch.to(device)
            
            # Forward pass
            train_outputs = model(train_batch)
            test_outputs = model(test_batch)
            
            train_probs = torch.sigmoid(train_outputs).cpu().numpy()
            test_probs = torch.sigmoid(test_outputs).cpu().numpy()
            
            print(f"Train batch predictions:")
            print(f"  Labels: {train_labels.numpy().flatten()}")
            print(f"  Probs:  {train_probs.flatten()}")
            print(f"  Preds:  {(train_probs > 0.5).astype(int).flatten()}")
            
            print(f"\nTest batch predictions:")
            print(f"  Labels: {test_labels.numpy().flatten()}")
            print(f"  Probs:  {test_probs.flatten()}")
            print(f"  Preds:  {(test_probs > 0.5).astype(int).flatten()}")
    
    # 6. Label distribution check
    print("\n6. LABEL DISTRIBUTION")
    print("-" * 40)
    
    train_labels = raw_train_dataset.segment_labels
    test_labels = raw_test_dataset.segment_labels
    
    print(f"Train labels:")
    print(f"  Class 0 (Resting): {np.sum(train_labels == 0)} ({np.mean(train_labels == 0)*100:.1f}%)")
    print(f"  Class 1 (Active):  {np.sum(train_labels == 1)} ({np.mean(train_labels == 1)*100:.1f}%)")
    
    print(f"\nTest labels:")
    print(f"  Class 0 (Resting): {np.sum(test_labels == 0)} ({np.mean(test_labels == 0)*100:.1f}%)")
    print(f"  Class 1 (Active):  {np.sum(test_labels == 1)} ({np.mean(test_labels == 1)*100:.1f}%)")
    
    print(f"\n{'='*60}")
    print("DEBUGGING COMPLETE")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    debug_data_consistency(exp_number=1)
    debug_data_consistency(exp_number=2)