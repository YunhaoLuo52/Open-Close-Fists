# import os
# import numpy as np
# import pandas as pd
# import torch
# from torch.utils.data import Dataset

# class TwoChannelEEGDataset(Dataset):
#     def __init__(self, data_dir, run_number, exp_number=1, window_size=256, overlap=0.5, task="openclosefists"):
#         """
#         Create windowed segments from continuous EEG data
#         """
#         self.window_size = window_size
#         self.overlap = overlap

#         # Load data
#         tp9_file = f"{task}_run{run_number}_TP9.csv"
#         tp10_file = f"{task}_run{run_number}_TP10.csv"
#         label_file = f"{task}_run{run_number}_label.csv"

#         exp_dir = os.path.join(data_dir, f"exp_{exp_number}")

#         tp9_data = pd.read_csv(os.path.join(exp_dir, tp9_file), header=None).values
#         tp10_data = pd.read_csv(os.path.join(exp_dir, tp10_file), header=None).values
#         labels = pd.read_csv(os.path.join(exp_dir, label_file), header=None).values.flatten()

#         # Create windowed segments
#         self.segments = []
#         self.segment_labels = []

#         for i, (tp9_sample, tp10_sample, label) in enumerate(zip(tp9_data, tp10_data, labels)):
#             data_length = len(tp9_sample)
            
#             # If window size equals data length, use original data without windowing
#             if window_size == data_length:
#                 # Stack the two channels
#                 segment = np.stack([tp9_sample, tp10_sample], axis=0)  # Shape: [2, data_length]
#                 self.segments.append(segment)
#                 self.segment_labels.append(label)
#             else:
#                 # Create overlapping windows
#                 step = int(window_size * (1 - overlap))
                
#                 for start in range(0, data_length - window_size + 1, step):
#                     end = start + window_size

#                     # Stack the two channels
#                     segment = np.stack([
#                         tp9_sample[start:end],
#                         tp10_sample[start:end]
#                     ], axis=0)  # Shape: [2, window_size]

#                     self.segments.append(segment)
#                     self.segment_labels.append(label)

#         self.segments = np.array(self.segments)
#         self.segment_labels = np.array(self.segment_labels)

#         print(f"Created {len(self.segments)} segments from {task} run {run_number}")
#         print(f"Segment shape: {self.segments.shape}")
#         print(f"Label distribution: {np.bincount(self.segment_labels.astype(int))}")

#     def __len__(self):
#         return len(self.segments)

#     def __getitem__(self, idx):
#         segment = torch.FloatTensor(self.segments[idx])
#         label = torch.tensor(self.segment_labels[idx], dtype=torch.float)
#         return segment, label

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

class TwoChannelEEGDataset(Dataset):
    # def __init__(self, data_dir, run_number, exp_number=1, window_size=256, overlap=0.5, 
    #              task="openclosefists", normalize_method='none', debug=True):
    #     """
    #     Fixed EEG dataset with proper handling of pre-processed data
    #     """
    #     self.window_size = window_size
    #     self.overlap = overlap
    #     self.normalize_method = normalize_method
    #     self.segments = []
    #     self.segment_labels = []
    #     self.segment_metadata = []
        
    #     # Load data
    #     tp9_file = f"{task}_run{run_number}_TP9.csv"
    #     tp10_file = f"{task}_run{run_number}_TP10.csv"
    #     label_file = f"{task}_run{run_number}_label.csv"
        
    #     exp_dir = os.path.join(data_dir, f"exp_{exp_number}")
        
    #     tp9_data = pd.read_csv(os.path.join(exp_dir, tp9_file), header=None).values
    #     tp10_data = pd.read_csv(os.path.join(exp_dir, tp10_file), header=None).values
    #     labels = pd.read_csv(os.path.join(exp_dir, label_file), header=None).values.flatten()
        
    #     if debug:
    #         print(f"=== ORIGINAL DATA ANALYSIS ===")
    #         print(f"TP9 shape: {tp9_data.shape}")
    #         print(f"TP10 shape: {tp10_data.shape}")
    #         print(f"Labels shape: {labels.shape}")
    #         print(f"TP9 range: [{np.min(tp9_data):.2e}, {np.max(tp9_data):.2e}]")
    #         print(f"TP10 range: [{np.min(tp10_data):.2e}, {np.max(tp10_data):.2e}]")
            
    #         # Check per-class statistics
    #         for class_label in np.unique(labels):
    #             class_mask = labels == class_label
    #             class_name = "Resting" if class_label == 0 else "Open/Close Fist"
                
    #             tp9_class = tp9_data[class_mask]
    #             tp10_class = tp10_data[class_mask]
                
    #             print(f"\n{class_name} (label {class_label}):")
    #             print(f"  TP9 mean: {np.mean(tp9_class):.2e}")
    #             print(f"  TP10 mean: {np.mean(tp10_class):.2e}")
    #             print(f"  TP9 std: {np.std(tp9_class):.2e}")
    #             print(f"  TP10 std: {np.std(tp10_class):.2e}")
        
    #     # Apply normalization if requested
    #     if normalize_method == 'global_std':
    #         # Normalize by global standard deviation across all data
    #         global_std = np.std(np.concatenate([tp9_data.flatten(), tp10_data.flatten()]))
    #         tp9_data = tp9_data / global_std
    #         tp10_data = tp10_data / global_std
    #         if debug:
    #             print(f"\nApplied global std normalization (std={global_std:.2e})")
                
    #     elif normalize_method == 'channel_wise':
    #         # Normalize each channel independently
    #         tp9_std = np.std(tp9_data)
    #         tp10_std = np.std(tp10_data)
    #         tp9_data = tp9_data / tp9_std
    #         tp10_data = tp10_data / tp10_std
    #         if debug:
    #             print(f"\nApplied channel-wise normalization")
    #             print(f"  TP9 std: {tp9_std:.2e}")
    #             print(f"  TP10 std: {tp10_std:.2e}")
        
    #     elif normalize_method == 'robust':
    #         # Use robust normalization (median, IQR)
    #         tp9_median = np.median(tp9_data)
    #         tp10_median = np.median(tp10_data)
    #         tp9_iqr = np.percentile(tp9_data, 75) - np.percentile(tp9_data, 25)
    #         tp10_iqr = np.percentile(tp10_data, 75) - np.percentile(tp10_data, 25)
            
    #         tp9_data = (tp9_data - tp9_median) / tp9_iqr
    #         tp10_data = (tp10_data - tp10_median) / tp10_iqr
    #         if debug:
    #             print(f"\nApplied robust normalization")
        
    #     # Create windowed segments
    #     self.segments = []
    #     self.segment_labels = []
    #     self.segment_metadata = []
        
    #     for i, (tp9_sample, tp10_sample, label) in enumerate(zip(tp9_data, tp10_data, labels)):
    #         data_length = len(tp9_sample)
            
    #         # If window size equals data length, use original data without windowing
    #         if window_size == data_length:
    #             segment = np.stack([tp9_sample, tp10_sample], axis=0)
    #             self.segments.append(segment)
    #             self.segment_labels.append(label)
    #             self.segment_metadata.append({
    #                 'original_trial': i,
    #                 'window_start': 0,
    #                 'window_end': data_length,
    #                 'run': run_number
    #             })
    #         else:
    #             # Create overlapping windows
    #             step = int(window_size * (1 - overlap))
                
    #             for start in range(0, data_length - window_size + 1, step):
    #                 end = start + window_size
                    
    #                 segment = np.stack([
    #                     tp9_sample[start:end],
    #                     tp10_sample[start:end]
    #                 ], axis=0)
                    
    #                 self.segments.append(segment)
    #                 self.segment_labels.append(label)
    #                 self.segment_metadata.append({
    #                     'original_trial': i,
    #                     'window_start': start,
    #                     'window_end': end,
    #                     'run': run_number
    #                 })
        
    #     self.segments = np.array(self.segments)
    #     self.segment_labels = np.array(self.segment_labels)
        
    #     if debug:
    #         print(f"\n=== FINAL SEGMENTS ===")
    #         print(f"Created {len(self.segments)} segments from {task} run {run_number}")
    #         print(f"Segment shape: {self.segments.shape}")
    #         print(f"Label distribution: {np.bincount(self.segment_labels.astype(int))}")
            
    #         # Check if the problem persists
    #         self._analyze_class_separability()


    def __init__(self, data_dir, run_number, exp_number=1, window_size=1025, overlap=0.5, 
                 task="openclosefists", debug=True):

        self.window_size = window_size
        self.overlap = overlap
        self.segments = []
        self.segment_labels = []
        self.segment_metadata = []

        if isinstance(run_number, int):
            run_number = [run_number]  # convert to list for uniform processing

        for run in run_number:
            self._load_run(data_dir, run, exp_number, task, debug)

        self.segments = np.array(self.segments)
        self.segment_labels = np.array(self.segment_labels)

        if debug:
            print(f"\n=== FINAL SEGMENTS ===")
            print(f"Created {len(self.segments)} segments from runs {run_number}")
            print(f"Segment shape: {self.segments.shape}")
            print(f"Label distribution: {np.bincount(self.segment_labels.astype(int))}")

            self._analyze_class_separability()

    def _load_run(self, data_dir, run_number, exp_number, task, debug):
        exp_dir = os.path.join(data_dir, f"exp_{exp_number}")
        tp9_file = f"{task}_run{run_number}_TP9.csv"
        tp10_file = f"{task}_run{run_number}_TP10.csv"
        label_file = f"{task}_run{run_number}_label.csv"

        tp9_data = pd.read_csv(os.path.join(exp_dir, tp9_file), header=None).values
        tp10_data = pd.read_csv(os.path.join(exp_dir, tp10_file), header=None).values
        labels = pd.read_csv(os.path.join(exp_dir, label_file), header=None).values.flatten()

        if debug:
            print(f"\n=== RUN {run_number} DATA ANALYSIS ===")
            print(f"TP9 shape: {tp9_data.shape}")
            print(f"TP10 shape: {tp10_data.shape}")
            print(f"Labels shape: {labels.shape}")


        # Segment creation
        for i, (tp9_sample, tp10_sample, label) in enumerate(zip(tp9_data, tp10_data, labels)):
            data_length = len(tp9_sample)
            if self.window_size == data_length:
                segment = np.stack([tp9_sample, tp10_sample], axis=0)
                self.segments.append(segment)
                self.segment_labels.append(label)
                self.segment_metadata.append({
                    'original_trial': i,
                    'window_start': 0,
                    'window_end': data_length,
                    'run': run_number
                })
            else:
                step = int(self.window_size * (1 - self.overlap))
                for start in range(0, data_length - self.window_size + 1, step):
                    end = start + self.window_size
                    segment = np.stack([
                        tp9_sample[start:end],
                        tp10_sample[start:end]
                    ], axis=0)
                    self.segments.append(segment)
                    self.segment_labels.append(label)
                    self.segment_metadata.append({
                        'original_trial': i,
                        'window_start': start,
                        'window_end': end,
                        'run': run_number
                    })








    
    def _analyze_class_separability(self):
        """Analyze how easily separable the classes are"""
        print(f"\n=== CLASS SEPARABILITY ANALYSIS ===")
        
        # Calculate simple statistics per class
        class_stats = {}
        for class_label in np.unique(self.segment_labels):
            class_mask = self.segment_labels == class_label
            class_segments = self.segments[class_mask]
            
            class_stats[class_label] = {
                'mean_amplitude': np.mean(class_segments),
                'std_amplitude': np.std(class_segments),
                'mean_ch1': np.mean(class_segments[:, 0, :]),
                'mean_ch2': np.mean(class_segments[:, 1, :]),
                'std_ch1': np.std(class_segments[:, 0, :]),
                'std_ch2': np.std(class_segments[:, 1, :])
            }
        
        for class_label, stats in class_stats.items():
            class_name = "Resting" if class_label == 0 else "Open/Close Fist"
            print(f"\n{class_name}:")
            print(f"  Overall mean: {stats['mean_amplitude']:.2e}")
            print(f"  Channel 1 mean: {stats['mean_ch1']:.2e}")
            print(f"  Channel 2 mean: {stats['mean_ch2']:.2e}")
        
        # Check if classes are trivially separable by simple statistics
        if len(class_stats) == 2:
            class_0_mean = class_stats[0]['mean_amplitude']
            class_1_mean = class_stats[1]['mean_amplitude']
            mean_difference = abs(class_1_mean - class_0_mean)
            
            combined_std = np.sqrt(class_stats[0]['std_amplitude']**2 + class_stats[1]['std_amplitude']**2)
            
            # Cohen's d effect size
            cohens_d = mean_difference / combined_std if combined_std > 0 else float('inf')
            
            print(f"\nSeparability metrics:")
            print(f"  Mean difference: {mean_difference:.2e}")
            print(f"  Cohen's d effect size: {cohens_d:.2f}")
            
            if cohens_d > 2.0:
                print("⚠️  VERY HIGH effect size - classes are trivially separable!")
                print("   This explains the perfect classification performance.")
            elif cohens_d > 0.8:
                print("⚠️  HIGH effect size - classes may be too easily separable")
            else:
                print("✓  Reasonable effect size for classification")
    
    def plot_class_comparison(self, n_samples=3):
        """Plot samples from each class for comparison"""
        fig, axes = plt.subplots(2, n_samples, figsize=(15, 8))
        
        for class_label in [0, 1]:
            class_name = "Resting" if class_label == 0 else "Open/Close Fist"
            class_indices = np.where(self.segment_labels == class_label)[0]
            
            for i in range(min(n_samples, len(class_indices))):
                sample_idx = class_indices[i]
                sample = self.segments[sample_idx]
                
                axes[class_label, i].plot(sample[0], label='Channel 1', alpha=0.7)
                axes[class_label, i].plot(sample[1], label='Channel 2', alpha=0.7)
                axes[class_label, i].set_title(f'{class_name} - Sample {i+1}')
                axes[class_label, i].legend()
                axes[class_label, i].grid(True)
                
                # Add statistics
                ch1_mean = np.mean(sample[0])
                ch2_mean = np.mean(sample[1])
                axes[class_label, i].text(0.02, 0.98, 
                                        f'CH1: {ch1_mean:.2e}\nCH2: {ch2_mean:.2e}',
                                        transform=axes[class_label, i].transAxes,
                                        verticalalignment='top',
                                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.show()
    
    def create_proper_train_test_split(self, test_size=0.3, method='trial_based'):
        """Create proper train/test split to avoid data leakage"""
        if method == 'trial_based':
            # Split based on original trials, not windows
            unique_trials = np.unique([meta['original_trial'] for meta in self.segment_metadata])
            np.random.shuffle(unique_trials)
            
            n_test_trials = int(len(unique_trials) * test_size)
            test_trials = set(unique_trials[:n_test_trials])
            
            train_indices = []
            test_indices = []
            
            for i, meta in enumerate(self.segment_metadata):
                if meta['original_trial'] in test_trials:
                    test_indices.append(i)
                else:
                    train_indices.append(i)
            
            return train_indices, test_indices
        
        elif method == 'temporal':
            # Split based on time (earlier vs later segments)
            n_test = int(len(self.segments) * test_size)
            indices = np.arange(len(self.segments))
            
            train_indices = indices[:-n_test]
            test_indices = indices[-n_test:]
            
            return train_indices, test_indices
    
    def __len__(self):
        return len(self.segments)
    
    def __getitem__(self, idx):
        segment = torch.FloatTensor(self.segments[idx])
        label = torch.tensor(self.segment_labels[idx], dtype=torch.float)
        return segment, label

# Usage for debugging your data:
"""
# Load with different normalization methods
dataset_raw = FixedTwoChannelEEGDataset(
    data_dir="your_data_dir", 
    run_number=1, 
    normalize_method='none',
    debug=True
)

dataset_normalized = FixedTwoChannelEEGDataset(
    data_dir="your_data_dir", 
    run_number=1, 
    normalize_method='global_std',
    debug=True
)

# Visualize the problem
dataset_raw.plot_class_comparison()

# Create proper train/test split
train_indices, test_indices = dataset_raw.create_proper_train_test_split(method='trial_based')
"""