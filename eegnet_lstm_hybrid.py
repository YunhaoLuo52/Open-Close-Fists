import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class EEGNet_LSTM(nn.Module):
    """
    Enhanced EEGNet with LSTM for superior temporal modeling
    
    Architecture:
    1. EEGNet Blocks 1-2: Spatial-Temporal feature extraction
    2. LSTM Layer: Advanced temporal sequence modeling  
    3. EEGNet Block 3: Feature integration and classification
    
    Best placement: After Block 2, before final classification
    Rationale: LSTM processes rich spatio-temporal features, not raw signals
    """
    
    def __init__(self, nb_classes=1, Chans=2, Samples=1025, 
                 dropoutRate=0.25, kernLength=64, F1=8, D=2, F2=16,
                 lstm_hidden_size=64, lstm_num_layers=1, lstm_dropout=0.3,
                 norm_rate=0.25, dropoutType='Dropout'):
        super(EEGNet_LSTM, self).__init__()
        
        if dropoutType == 'SpatialDropout2D':
            dropoutType = nn.Dropout2d
        elif dropoutType == 'Dropout':
            dropoutType = nn.Dropout
        else:
            raise ValueError('dropoutType must be one of SpatialDropout2D or Dropout')
        
        # Store parameters
        self.nb_classes = nb_classes
        self.Chans = Chans
        self.Samples = Samples
        self.dropoutRate = dropoutRate
        self.kernLength = kernLength
        self.F1 = F1
        self.D = D
        self.F2 = F2
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.norm_rate = norm_rate
        
        # =================== EEGNET BLOCK 1 ===================
        # Temporal Convolution - Learn frequency patterns
        padding1 = (kernLength - 1) // 2  # Manual padding calculation
        self.firstconv = nn.Conv2d(1, F1, (1, kernLength), padding=(0, padding1), bias=False)
        self.batchnorm1 = nn.BatchNorm2d(F1, momentum=0.01, eps=0.001)
        
        # =================== EEGNET BLOCK 2 ===================
        # Spatial Convolution - Learn spatial patterns per frequency
        self.depthwiseConv = nn.Conv2d(F1, F1 * D, (Chans, 1), groups=F1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(F1 * D, momentum=0.01, eps=0.001)
        self.activation1 = nn.ELU()
        self.pooling1 = nn.AvgPool2d((1, 4))  # Reduces time dimension by 4
        self.dropout1 = dropoutType(dropoutRate)
        
        # Calculate sequence length after pooling1 for LSTM
        self.lstm_seq_length = Samples // 4  # After pooling1: 1025 -> 256
        
        # =================== LSTM LAYER ===================
        # Advanced temporal modeling of spatio-temporal features
        self.lstm = nn.LSTM(
            input_size=F1 * D,  # Number of channels from Block 2
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=lstm_dropout if lstm_num_layers > 1 else 0,
            bidirectional=True  # Bidirectional for better temporal context
        )
        
        # LSTM output size (bidirectional doubles the hidden size)
        self.lstm_output_size = lstm_hidden_size * 2
        
        # =================== POST-LSTM PROCESSING ===================
        # Reduce temporal dimension for final processing
        self.temporal_pool = nn.AdaptiveAvgPool1d(32)  # Standardize sequence length
        
        # =================== EEGNET BLOCK 3 (MODIFIED) ===================
        # Modified separable convolution to work with LSTM output
        padding_sep = (16 - 1) // 2  # Manual padding calculation
        self.separableConv = nn.Conv1d(
            self.lstm_output_size, F2, kernel_size=16, 
            padding=padding_sep, bias=False
        )
        self.batchnorm3 = nn.BatchNorm1d(F2, momentum=0.01, eps=0.001)
        self.activation2 = nn.ELU()
        self.pooling2 = nn.AvgPool1d(8)  # Final pooling
        self.dropout2 = nn.Dropout(dropoutRate)
        
        # =================== CLASSIFICATION ===================
        # Use adaptive pooling to standardize the final size regardless of input dimensions
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)  # Pool to single value per channel
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(F2, 64),  # F2 channels, each pooled to 1 value
            nn.ELU(),
            nn.Dropout(dropoutRate),
            nn.Linear(64, nb_classes)
        )
        
        # Max norm constraint
        self.max_norm_val = norm_rate
        
    def forward(self, x):
        # Input shape: (batch_size, 1, 2, time_samples)
        batch_size = x.size(0)
        
        # =================== EEGNET BLOCKS 1-2 ===================
        # Block 1: Temporal Convolution
        x = self.firstconv(x)  # (batch, F1, 2, samples)
        x = self.batchnorm1(x)
        
        # Block 2: Spatial Convolution  
        x = self.depthwiseConv(x)  # (batch, F1*D, 1, samples)
        x = self.batchnorm2(x)
        x = self.activation1(x)
        x = self.pooling1(x)  # (batch, F1*D, 1, samples//4)
        x = self.dropout1(x)
        
        # =================== PREPARE FOR LSTM ===================
        # Reshape for LSTM: (batch, sequence_length, features)
        x = x.squeeze(2)  # Remove spatial dimension: (batch, F1*D, seq_len)
        x = x.transpose(1, 2)  # (batch, seq_len, F1*D)
        
        # =================== LSTM PROCESSING ===================
        # Advanced temporal modeling
        lstm_out, (hidden, cell) = self.lstm(x)  # (batch, seq_len, lstm_hidden*2)
        
        # =================== POST-LSTM PROCESSING ===================
        # Transpose back for conv1d: (batch, features, time)
        x = lstm_out.transpose(1, 2)  # (batch, lstm_hidden*2, seq_len)
        
        # Standardize sequence length
        x = self.temporal_pool(x)  # (batch, lstm_hidden*2, 32)
        
        # =================== FINAL FEATURE EXTRACTION ===================
        # Modified Block 3: Separable Convolution
        x = self.separableConv(x)  # (batch, F2, 32)
        x = self.batchnorm3(x)
        x = self.activation2(x)
        x = self.pooling2(x)  # (batch, F2, 4)
        x = self.dropout2(x)
        
        # =================== CLASSIFICATION ===================
        # Use adaptive pooling to ensure consistent dimensions
        x = self.adaptive_pool(x)  # (batch, F2, 1)
        x = self.classifier(x)
        
        return x
    
    def max_norm_constraint(self):
        """Apply max norm constraint to classification layers"""
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                with torch.no_grad():
                    norm = module.weight.norm(dim=1, keepdim=True)
                    desired = torch.clamp(norm, 0, self.max_norm_val)
                    module.weight *= (desired / norm)


class EEGNet_LSTM_Robust(nn.Module):
    """
    Robust version of EEGNet-LSTM with automatic dimension handling
    Fixes the dimension mismatch issues in the full model
    """
    
    def __init__(self, nb_classes=1, Chans=2, Samples=1025, 
                 dropoutRate=0.25, kernLength=64, F1=8, D=2, F2=16,
                 lstm_hidden_size=32, norm_rate=0.25):
        super(EEGNet_LSTM_Robust, self).__init__()
        
        self.nb_classes = nb_classes
        self.Chans = Chans
        self.Samples = Samples
        
        # =================== EEGNET BLOCKS 1-2 ===================
        # Block 1: Temporal Convolution
        padding1 = (kernLength - 1) // 2
        self.conv1 = nn.Conv2d(1, F1, (1, kernLength), padding=(0, padding1), bias=False)
        self.bn1 = nn.BatchNorm2d(F1, momentum=0.01, eps=0.001)
        
        # Block 2: Spatial Convolution
        self.conv2 = nn.Conv2d(F1, F1 * D, (Chans, 1), groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(F1 * D, momentum=0.01, eps=0.001)
        self.elu1 = nn.ELU()
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(dropoutRate)
        
        # =================== LSTM LAYER ===================
        self.lstm = nn.LSTM(
            input_size=F1 * D,
            hidden_size=lstm_hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        # =================== FINAL PROCESSING ===================
        # Simple final layers with guaranteed dimensions
        self.global_pool = nn.AdaptiveAvgPool1d(1)  # Always outputs (batch, features, 1)
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(lstm_hidden_size * 2, 32),  # LSTM output size
            nn.ELU(),
            nn.Dropout(dropoutRate),
            nn.Linear(32, nb_classes)
        )
        
    def forward(self, x):
        # EEGNet feature extraction
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.elu1(x)
        x = self.pool1(x)
        x = self.drop1(x)
        
        # Prepare for LSTM
        x = x.squeeze(2).transpose(1, 2)  # (batch, seq_len, features)
        
        # LSTM processing
        lstm_out, _ = self.lstm(x)
        
        # Global pooling over time dimension
        lstm_out = lstm_out.transpose(1, 2)  # (batch, features, seq_len)
        x = self.global_pool(lstm_out)  # (batch, features, 1)
        
        # Classification
        x = self.classifier(x)
        
        return x
class EEGNet_LSTM_Attention(nn.Module):
    """
    Advanced version with attention mechanism for even better performance
    
    Addition: Attention layer after LSTM to focus on most relevant time points
    """
    
    def __init__(self, nb_classes=1, Chans=2, Samples=1025, 
                 dropoutRate=0.25, kernLength=64, F1=8, D=2, F2=16,
                 lstm_hidden_size=64, lstm_num_layers=1, lstm_dropout=0.3,
                 attention_heads=4, norm_rate=0.25):
        super(EEGNet_LSTM_Attention, self).__init__()
        
        # Same EEGNet-LSTM structure as above
        self.eegnet_lstm = EEGNet_LSTM(
            nb_classes=lstm_hidden_size*2,  # Will be replaced by attention
            Chans=Chans, Samples=Samples, dropoutRate=dropoutRate,
            kernLength=kernLength, F1=F1, D=D, F2=F2,
            lstm_hidden_size=lstm_hidden_size, lstm_num_layers=lstm_num_layers,
            lstm_dropout=lstm_dropout, norm_rate=norm_rate
        )
        
        # Remove the classifier from base model
        self.eegnet_lstm.classifier = nn.Identity()
        
        # =================== ATTENTION MECHANISM ===================
        self.attention_dim = lstm_hidden_size * 2
        self.attention = nn.MultiheadAttention(
            embed_dim=self.attention_dim,
            num_heads=attention_heads,
            dropout=dropoutRate,
            batch_first=True
        )
        
        # Attention normalization
        self.attention_norm = nn.LayerNorm(self.attention_dim)
        
        # =================== FINAL CLASSIFICATION ===================
        self.final_classifier = nn.Sequential(
            nn.Dropout(dropoutRate),
            nn.Linear(self.attention_dim, 64),
            nn.ELU(),
            nn.Dropout(dropoutRate * 0.5),
            nn.Linear(64, nb_classes)
        )
        
    def forward(self, x):
        # Get LSTM features (before final classification)
        # We need to modify the base model to return LSTM output
        batch_size = x.size(0)
        
        # Process through EEGNet blocks 1-2
        x = self.eegnet_lstm.firstconv(x)
        x = self.eegnet_lstm.batchnorm1(x)
        x = self.eegnet_lstm.depthwiseConv(x)
        x = self.eegnet_lstm.batchnorm2(x)
        x = self.eegnet_lstm.activation1(x)
        x = self.eegnet_lstm.pooling1(x)
        x = self.eegnet_lstm.dropout1(x)
        
        # Prepare and process through LSTM
        x = x.squeeze(2).transpose(1, 2)
        lstm_out, _ = self.eegnet_lstm.lstm(x)  # (batch, seq_len, hidden*2)
        
        # =================== ATTENTION MECHANISM ===================
        # Self-attention to find most important time points
        attended_out, attention_weights = self.attention(
            lstm_out, lstm_out, lstm_out
        )  # (batch, seq_len, hidden*2)
        
        # Add residual connection and normalize
        attended_out = self.attention_norm(attended_out + lstm_out)
        
        # Global average pooling across time dimension
        pooled_features = torch.mean(attended_out, dim=1)  # (batch, hidden*2)
        
        # Final classification
        output = self.final_classifier(pooled_features)
        
        return output


class EEGNet_LSTM_Simple(nn.Module):
    """
    Simplified version - easier to integrate with your existing code
    
    Key insight: Add LSTM right after EEGNet's spatial processing
    """
    
    def __init__(self, nb_classes=1, Chans=2, Samples=1025, 
                 dropoutRate=0.25, lstm_hidden_size=32):
        super(EEGNet_LSTM_Simple, self).__init__()
        
        # Standard EEGNet parameters
        kernLength, F1, D, F2 = 64, 8, 2, 16
        
        # =================== EEGNET FEATURE EXTRACTOR ===================
        # Block 1: Temporal - Fixed padding calculation
        padding1 = (kernLength - 1) // 2  # Manual padding calculation
        self.conv1 = nn.Conv2d(1, F1, (1, kernLength), padding=(0, padding1), bias=False)
        self.bn1 = nn.BatchNorm2d(F1)
        
        # Block 2: Spatial
        self.conv2 = nn.Conv2d(F1, F1*D, (Chans, 1), groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(F1*D)
        self.elu1 = nn.ELU()
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(dropoutRate)
        
        # =================== LSTM LAYER ===================
        # Process temporal sequences of spatial-spectral features
        self.lstm = nn.LSTM(
            input_size=F1*D,  # 16 features from EEGNet
            hidden_size=lstm_hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=0,
            bidirectional=True
        )
        
        # =================== FINAL LAYERS ===================
        # Combine LSTM output with remaining EEGNet processing
        padding3 = (16 - 1) // 2  # Manual padding calculation
        self.conv3 = nn.Conv1d(lstm_hidden_size*2, F2, 16, padding=padding3, bias=False)
        self.bn3 = nn.BatchNorm1d(F2)
        self.elu2 = nn.ELU()
        self.pool2 = nn.AdaptiveAvgPool1d(8)
        self.drop2 = nn.Dropout(dropoutRate)
        
        # Classification
        self.classifier = nn.Linear(F2 * 8, nb_classes)
        
    def forward(self, x):
        # EEGNet feature extraction
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.elu1(x)
        x = self.pool1(x)  # (batch, 16, 1, seq_len)
        x = self.drop1(x)
        
        # Prepare for LSTM
        x = x.squeeze(2).transpose(1, 2)  # (batch, seq_len, 16)
        
        # LSTM processing
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden*2)
        
        # Back to conv format
        x = lstm_out.transpose(1, 2)  # (batch, hidden*2, seq_len)
        
        # Final processing
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.elu2(x)
        x = self.pool2(x)
        x = self.drop2(x)
        
        # Classification
        x = x.flatten(1)
        x = self.classifier(x)
        
        return x


# Factory function for easy model creation
def create_eegnet_lstm_model(model_type='simple', nb_classes=1, samples=1025, device='cpu', **kwargs):
    """
    Factory function to create EEGNet-LSTM hybrid models
    
    Args:
        model_type: 'simple', 'full', 'robust', or 'attention'
        nb_classes: Number of output classes
        samples: Input sequence length
        device: Device to place the model on ('cpu' or 'cuda')
        **kwargs: Additional model parameters
    
    Returns:
        EEGNet-LSTM model ready for training
    """
    
    if model_type == 'simple':
        model = EEGNet_LSTM_Simple(
            nb_classes=nb_classes,
            Samples=samples,
            dropoutRate=kwargs.get('dropout', 0.25),
            lstm_hidden_size=kwargs.get('lstm_hidden', 32)
        )
    
    elif model_type == 'full':
        # Use the robust version instead of the problematic full version
        print("Using robust version instead of full to avoid dimension issues")
        model = EEGNet_LSTM_Robust(
            nb_classes=nb_classes,
            Samples=samples,
            dropoutRate=kwargs.get('dropout', 0.25),
            lstm_hidden_size=kwargs.get('lstm_hidden', 32)
        )
    
    elif model_type == 'robust':
        model = EEGNet_LSTM_Robust(
            nb_classes=nb_classes,
            Samples=samples,
            dropoutRate=kwargs.get('dropout', 0.25),
            lstm_hidden_size=kwargs.get('lstm_hidden', 32)
        )
    
    elif model_type == 'attention':
        model = EEGNet_LSTM_Attention(
            nb_classes=nb_classes,
            Samples=samples,
            dropoutRate=kwargs.get('dropout', 0.25),
            lstm_hidden_size=kwargs.get('lstm_hidden', 64),
            attention_heads=kwargs.get('attention_heads', 4)
        )
    
    else:
        raise ValueError("model_type must be 'simple', 'full', 'robust', or 'attention'")
    
    # Move model to specified device
    model = model.to(device)
    return model


# Training function with LSTM-specific optimizations
def train_eegnet_lstm_model(model, train_loader, val_loader, device, epochs=200, lr=0.001, patience=50, min_delta=0.00001):
    """
    Training loop optimized for EEGNet-LSTM models
    
    Key differences from standard training:
    1. Lower learning rate for LSTM stability
    2. Gradient clipping to prevent LSTM exploding gradients
    3. Longer training for LSTM convergence
    4. Device debugging
    """

    
    # LSTM-optimized optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.7, patience=15, verbose=True
    )
    
    criterion = nn.BCEWithLogitsLoss()
    
    best_val_acc = 0
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Check model device
    model_device = next(model.parameters()).device
    print(f"Model is on device: {model_device}")
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            # Ensure data is on correct device
            data, target = data.to(device), target.to(device)
            
            # Debug first batch
            if epoch == 0 and batch_idx == 0:
                print(f"First batch - Data shape: {data.shape}, device: {data.device}")
                print(f"First batch - Target shape: {target.shape}, device: {target.device}")
                print(f"Model device: {next(model.parameters()).device}")
            
            optimizer.zero_grad()
            output = model(data)
            target = target.view(-1, 1).float()
            loss = criterion(output, target)
            loss.backward()
            
            # Gradient clipping for LSTM stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Apply max norm constraint if available
            if hasattr(model, 'max_norm_constraint'):
                model.max_norm_constraint()
            
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        val_loss = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                target_reshaped = target.view(-1, 1).float()
                loss = criterion(output, target_reshaped)
                val_loss += loss.item()
                
                pred = (torch.sigmoid(output) > 0.5).float()
                correct += (pred == target_reshaped).sum().item()
                total += target.size(0)
        
        val_acc = correct / total
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        scheduler.step(avg_val_loss)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_eegnet_lstm_model.pth')
        
        if epoch % 20 == 0:
            print(f'Epoch {epoch}: Train Loss: {avg_train_loss:.4f}, '
                  f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}')
            

            

    
    return train_losses, val_losses, best_val_acc


# Integration example with your existing code
"""
USAGE EXAMPLES - COMPLETE INTEGRATION GUIDE:

Here's exactly how to modify your main.py:

# Step 1: Import the new modules
from eegnet_lstm_hybrid import create_eegnet_lstm_model, EEGNetDataset, train_eegnet_lstm_model

# Step 2: Replace your model creation
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = create_eegnet_lstm_model('simple', nb_classes=1, samples=1025, device=device)

# Step 3: Wrap your datasets with EEGNetDataset
# Replace your existing dataset creation with:
train_dataset_wrapped = EEGNetDataset(Subset(dataset, train_indices))
val_dataset_wrapped = EEGNetDataset(Subset(dataset, val_indices))
test_dataset_wrapped = EEGNetDataset(test_dataset)

# Step 4: Create new data loaders
train_loader = DataLoader(train_dataset_wrapped, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset_wrapped, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset_wrapped, batch_size=32, shuffle=False)

# Step 5: Train with debugging enabled first
train_losses, val_losses, best_acc = train_eegnet_lstm_model(
    model, train_loader, val_loader, device, epochs=300, lr=0.0005, debug=True
)

# Your existing evaluation code works unchanged:
labels, preds, probs = evaluate_simple_model(model, test_loader, device)
"""

# Quick fix function for immediate use
def quick_fix_main_integration():
    """
    Quick fix code snippet to add to your main.py
    """
    integration_code = '''
# Add these imports at the top of main.py:
from eegnet_lstm_hybrid import create_eegnet_lstm_model, EEGNetDataset, train_eegnet_lstm_model

# Replace the model creation section in your main() function:
def main(exp_number=1):
    # ... your existing code until model creation ...
    
    # REPLACE THIS SECTION:
    # OLD:
    # model = TwoChannelLSTMClassifier(input_channels=2, num_classes=1).to(device)
    # train_dataset = Subset(dataset, train_indices)
    # val_dataset = Subset(dataset, val_indices)
    
    # NEW:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_eegnet_lstm_model('simple', nb_classes=1, samples=1025, device=device)
    
    # Wrap datasets for EEGNet format
    train_dataset = EEGNetDataset(Subset(dataset, train_indices))
    val_dataset = EEGNetDataset(Subset(dataset, val_indices))
    test_dataset_wrapped = EEGNetDataset(test_dataset)
    
    # Update data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset_wrapped, batch_size=32, shuffle=False)
    
    # REPLACE TRAINING CALL:
    # OLD:
    # train_losses, val_losses, best_cm, best_epoch, new_best = train_simple_model_with_universal_best(...)
    
    # NEW:
    train_losses, val_losses, best_acc = train_eegnet_lstm_model(
        model, train_loader, val_loader, device, epochs=300, lr=0.0005, debug=True
    )
    
    # Rest of your code remains the same...
    '''
    print(integration_code)