import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class EEGNet(nn.Module):
    """
    EEGNet implementation optimized for 2-channel motor imagery classification
    
    Based on: Lawhern et al. "EEGNet: a compact convolutional neural network 
    for EEG-based brain–computer interfaces" (2018)
    
    Optimized for:
    - 2 channels (TP9, TP10)
    - Motor imagery tasks (open/close fists, feet)
    - Input shape: (batch_size, 1, 2, time_samples)
    """
    
    def __init__(self, nb_classes=1, Chans=2, Samples=1025, 
                 dropoutRate=0.25, kernLength=64, F1=8, D=2, F2=16,
                 norm_rate=0.25, dropoutType='Dropout'):
        super(EEGNet, self).__init__()
        
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
        self.norm_rate = norm_rate
        
        # Block 1: Temporal Convolution
        # Learn frequency-specific temporal patterns
        padding1 = (kernLength - 1) // 2  # Manual padding calculation
        self.firstconv = nn.Conv2d(1, F1, (1, kernLength), padding=(0, padding1), bias=False)
        self.batchnorm1 = nn.BatchNorm2d(F1, momentum=0.01, eps=0.001)
        
        # Block 2: Spatial Convolution (Depthwise)
        # Learn spatial patterns for each temporal feature
        self.depthwiseConv = nn.Conv2d(F1, F1 * D, (Chans, 1), groups=F1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(F1 * D, momentum=0.01, eps=0.001)
        self.activation1 = nn.ELU()
        self.pooling1 = nn.AvgPool2d((1, 4))
        self.dropout1 = dropoutType(dropoutRate)
        
        # Block 3: Separable Convolution
        # Efficiently combine features
        self.separableConv = nn.Conv2d(F1 * D, F2, (1, 16), padding='same', bias=False)
        self.batchnorm3 = nn.BatchNorm2d(F2, momentum=0.01, eps=0.001)
        self.activation2 = nn.ELU()
        self.pooling2 = nn.AvgPool2d((1, 8))
        self.dropout2 = dropoutType(dropoutRate)
        
        # Calculate the size after convolutions and pooling
        # For input (1, 2, 1025): after pooling1 (/4) and pooling2 (/8) = 1025/32 ≈ 32
        self.final_conv_length = Samples // 32
        
        # Classification layer
        self.flatten = nn.Flatten()
        self.classify = nn.Linear(F2 * self.final_conv_length, nb_classes)
        
        # Apply max norm constraint to classification layer
        self.max_norm_val = norm_rate
        
    def forward(self, x):
        # Input shape: (batch_size, 1, 2, time_samples)
        
        # Block 1: Temporal Convolution
        x = self.firstconv(x)
        x = self.batchnorm1(x)
        
        # Block 2: Spatial Convolution
        x = self.depthwiseConv(x)
        x = self.batchnorm2(x)
        x = self.activation1(x)
        x = self.pooling1(x)
        x = self.dropout1(x)
        
        # Block 3: Separable Convolution
        x = self.separableConv(x)
        x = self.batchnorm3(x)
        x = self.activation2(x)
        x = self.pooling2(x)
        x = self.dropout2(x)
        
        # Flatten and classify
        x = self.flatten(x)
        x = self.classify(x)
        
        return x
    
    def max_norm_constraint(self):
        """Apply max norm constraint to the final classification layer"""
        if hasattr(self.classify, 'weight'):
            with torch.no_grad():
                norm = self.classify.weight.norm(dim=1, keepdim=True)
                desired = torch.clamp(norm, 0, self.max_norm_val)
                self.classify.weight *= (desired / norm)


class EEGNet_MultiClass(nn.Module):
    """
    EEGNet variant for multi-class classification
    Use this if you want to classify: Rest, Open Fist, Close Fist, Feet (4 classes)
    """
    
    def __init__(self, nb_classes=4, Chans=2, Samples=1025, 
                 dropoutRate=0.25, kernLength=64, F1=8, D=2, F2=16):
        super(EEGNet_MultiClass, self).__init__()
        
        # Same architecture as binary EEGNet
        self.eegnet_features = EEGNet(nb_classes=F2*Samples//32, Chans=Chans, 
                                     Samples=Samples, dropoutRate=dropoutRate,
                                     kernLength=kernLength, F1=F1, D=D, F2=F2)
        
        # Remove the classification layer from base EEGNet
        self.eegnet_features.classify = nn.Identity()
        
        # Custom multi-class classification head
        self.final_conv_length = Samples // 32
        self.classifier = nn.Sequential(
            nn.Dropout(dropoutRate),
            nn.Linear(F2 * self.final_conv_length, 64),
            nn.ELU(),
            nn.Dropout(dropoutRate * 0.5),
            nn.Linear(64, nb_classes)
        )
        
    def forward(self, x):
        # Extract features using EEGNet backbone
        x = self.eegnet_features(x)
        # Classify with multi-class head
        x = self.classifier(x)
        return x


# Modified Dataset class to work with EEGNet input format
class EEGNetDataset(torch.utils.data.Dataset):
    """
    Dataset wrapper that formats data for EEGNet
    Converts your current data format to EEGNet expected format
    """
    
    def __init__(self, base_dataset):
        """
        Args:
            base_dataset: Your existing TwoChannelEEGDataset
        """
        self.base_dataset = base_dataset
        
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        segment, label = self.base_dataset[idx]
        
        # Your current format: (2, time_samples)
        # EEGNet expected format: (1, 2, time_samples)
        
        # Add batch dimension for channels
        eegnet_segment = segment.unsqueeze(0)  # Shape: (1, 2, time_samples)
        
        return eegnet_segment, label


# Training modifications for EEGNet
class EEGNetTrainer:
    """
    Training utilities specific to EEGNet
    """
    
    @staticmethod
    def apply_max_norm_constraint(model):
        """Apply max norm constraint during training"""
        if hasattr(model, 'max_norm_constraint'):
            model.max_norm_constraint()
    
    @staticmethod
    def get_eegnet_optimizer(model, lr=0.001, weight_decay=1e-4):
        """Recommended optimizer settings for EEGNet"""
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    @staticmethod
    def get_eegnet_scheduler(optimizer):
        """Recommended learning rate scheduler for EEGNet"""
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )


def create_eegnet_model(task_type='binary', num_classes=1, samples=1025):
    """
    Factory function to create appropriate EEGNet model
    
    Args:
        task_type: 'binary' for your current setup, 'multiclass' for extended classification
        num_classes: 1 for binary classification, >1 for multiclass
        samples: number of time samples (your current: 1025)
    
    Returns:
        EEGNet model ready for training
    """
    
    if task_type == 'binary':
        model = EEGNet(
            nb_classes=num_classes,
            Chans=2,  # TP9, TP10
            Samples=samples,
            dropoutRate=0.25,  # Lower dropout for small dataset
            kernLength=64,     # Good for 256 Hz sampling rate
            F1=8,              # Number of temporal filters
            D=2,               # Depth multiplier
            F2=16              # Number of separable filters
        )
    elif task_type == 'multiclass':
        model = EEGNet_MultiClass(
            nb_classes=num_classes,
            Chans=2,
            Samples=samples,
            dropoutRate=0.25,
            kernLength=64,
            F1=8,
            D=2,
            F2=16
        )
    else:
        raise ValueError("task_type must be 'binary' or 'multiclass'")
    
    return model


def train_eegnet_model(model, train_loader, val_loader, device, epochs=100, lr=0.001):
    """
    Modified training loop for EEGNet with max norm constraint
    """
    
    # EEGNet-specific optimizer
    optimizer = EEGNetTrainer.get_eegnet_optimizer(model, lr=lr)
    scheduler = EEGNetTrainer.get_eegnet_scheduler(optimizer)
    
    # Your existing criterion (works fine with EEGNet)
    criterion = nn.BCEWithLogitsLoss()
    
    best_val_acc = 0
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            target = target.view(-1, 1).float()
            loss = criterion(output, target)
            loss.backward()
            
            # Apply max norm constraint (EEGNet specific)
            EEGNetTrainer.apply_max_norm_constraint(model)
            
            optimizer.step()
            train_loss += loss.item()
        
        # Validation (same as your current code)
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
            torch.save(model.state_dict(), 'best_eegnet_model.pth')
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch}: Train Loss: {avg_train_loss:.4f}, '
                  f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}')
    
    return train_losses, val_losses, best_val_acc
