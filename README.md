# EEG Motor Classification

A comprehensive deep learning framework for classifying motor tasks from EEG signals using Muse headband data. The system supports binary classification (rest vs motor ), multi-class classification (rest vs fists vs feet), and hierarchical two-stage classification approaches.

## Overview

This project implements EEGNet architecture for classifying motor tasks from 2-channel EEG data (TP9, TP10). The system is designed for brain-computer interface applications and motor research.

### Key Features

- **Multiple Classification Approaches**:
  - Binary classification: Rest vs Fists, Rest vs Feet
  - Multi-class classification: Rest vs Fists vs Feet
  - Two-stage hierarchical classification: (Rest vs Motor) → (Fists vs Feet)

- **Robust Data Pipeline**:
  - Flexible windowing with customizable overlap
  - Optional data normalization with parameter consistency
  - Proper train/validation/test splits to prevent data leakage
  - Support for cross-session validation

- **EEGNet Architecture**:
  - Optimized for 2-channel motor classification
  - Temporal and spatial convolutions with depthwise separable layers
  - Max-norm regularization and dropout for stability

## Project Structure

```
Open-Close/
├── dataset.py              # Data loading and preprocessing classes
├── model.py                # EEGNet model architectures
├── training.py             # Training loops and evaluation functions
├── evaluation.py           # Comprehensive evaluation and visualization
├── utils.py                # Utility functions and reproducibility helpers
├── main.py                 # Basic binary classification training
├── main_multiclass.py      # Multi-class classification training
├── main_two_stage.py       # Two-stage hierarchical classification
├── fists_feet_cls.py       # Combined multi-class training script
├── test.py                 # Basic model testing
├── test_multiclass.py      # Multi-class model testing
├── models/                 # Saved model weights and configurations
└── normalization_params/  # Saved normalization parameters
```


### Prerequisites

```bash
pip install torch torchvision numpy pandas matplotlib seaborn scikit-learn
```

### Basic Usage

1. **Prepare your data**: Organize EEG data in the expected directory structure:
   ```
   data_dir/
   ├── exp_1/  # Fists experiment
   │   ├── openclosefists_run1_TP9.csv
   │   ├── openclosefists_run1_TP10.csv
   │   ├── openclosefists_run1_label.csv
   │   └── ...
   └── exp_2/  # Feet experiment
       ├── openclosefeet_run1_TP9.csv
       ├── openclosefeet_run1_TP10.csv
       ├── openclosefeet_run1_label.csv
       └── ...
   ```

2. **Train binary classifiers**:
   ```bash
   python main.py
   ```

3. **Train multi-class classifier**:
   ```bash
   python main_multiclass.py
   ```

4. **Test models**:
   ```bash
   python test.py
   ```

## Classification Approaches

### 1. Binary Classification
Separate models for each motor task:
- **Fists Model**: Rest (0) vs Open/Close Fists (1)
- **Feet Model**: Rest (0) vs Open/Close Feet (1)


### 2. Multi-Class Classification
Single model classifying three states simultaneously:
- Class 0: Resting
- Class 1: Fists motor
- Class 2: Feet motor


### 3. Two-Stage Classification
Hierarchical approach for potentially better performance:
- **Stage 1**: Rest (0) vs Motor (1)
- **Stage 2**: Fists (0) vs Feet (1) [only for motor samples]

## Model Architecture

### EEGNet Implementation
Based on Lawhern et al. (2018), optimized for 2-channel motor:

**Architecture Details**:
- **Block 1**: Temporal convolution (8 filters, kernel=64)
- **Block 2**: Spatial depthwise convolution (16 filters)
- **Block 3**: Separable convolution (16 filters)
- **Regularization**: Dropout (0.25), max-norm constraint
- **Output**: Sigmoid (binary) or Softmax (multiclass)

## Evaluation and Visualization

### Comprehensive Metrics
- Accuracy, Precision, Recall, F1-Score
- ROC curves and AUC scores
- Confusion matrices (normalized and raw)
- Per-class performance analysis

### Visualizations
Generated plots include:
- Training/validation curves
- Confusion matrices
- ROC curves
- Prediction probability distributions
- Misclassification analysis

## Performance Optimization

### Training Best Practices
1. **Data Normalization**: Use normalization for consistent performance
2. **Cross-Session Validation**: Test on different recording sessions
3. **Proper Splits**: Use trial-based splitting to prevent overfitting
4. **Class Balancing**: Automatic class weight computation for imbalanced data

## Testing and Validation

### Cross-Session Testing
```python
# Test on different session
test_dataset = TwoChannelEEGDataset(
    data_dir="path/to/session2",  # Different session
    normalize=True,
    normalization_params=train_params  # Use training normalization
)
```

## Output Files

### Model Files
- `best_{experiment}_eegnet_model.pth`: Best model weights
- `models/{experiment}_model_config.json`: Model configuration
- `normalization_params/{experiment}_norm_params.json`: Normalization parameters

### Results
- `results/multiclass_results.png`: Comprehensive performance plots
- `results/multiclass_report.txt`: Detailed text report
- `{experiment}_train_result.png`: Training curves and performance
- `{experiment}_test_result.png`: Cllasification performance using saved model


## Common Issues

1. **Poor Test Performance**:
   - Check normalization parameter consistency
   - Verify data preprocessing pipeline

2. **CUDA Out of Memory**:
   - Reduce batch size in DataLoader
   - Use CPU if necessary: `device = torch.device("cpu")`

3. **Data Loading Errors**:
   - Verify file paths and naming conventions
   - Check CSV file formats (no headers expected)
   - Ensure consistent sampling rates

### Performance Tips
- Use GPU acceleration when available
- Enable normalization for optimal performance

## References
1. Lawhern, V. J., et al. (2018). EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces. Journal of Neural Engineering, 15(5), 056013.

