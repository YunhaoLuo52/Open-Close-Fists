import numpy as np
import torch
import torch.nn as nn
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report
import json
import os
from datetime import datetime

def train_simple_model(model, train_loader, val_loader, device, epochs=100, lr=1e-3):
    """Simple training loop"""

    # Calculate class weights
    train_labels = []
    for _, labels in train_loader:
        train_labels.extend(labels.numpy())

    class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
    pos_weight = torch.tensor(class_weights[1] / class_weights[0]).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    best_val_acc = 0
    train_losses = []
    val_losses = []
    best_epoch = 0
    best_confusion_matrix = None
    best_predictions = None
    best_targets = None

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
            optimizer.step()

            train_loss += loss.item()

        # Validation
        model.eval()
        correct = 0
        total = 0
        val_loss = 0
        all_predictions = []
        all_targets = []

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

                # Store predictions and targets for confusion matrix
                all_predictions.extend(pred.cpu().numpy().flatten())
                all_targets.extend(target_reshaped.cpu().numpy().flatten())

        val_acc = correct / total
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        scheduler.step(avg_val_loss)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch

            # Calculate and store confusion matrix for best performance
            best_predictions = np.array(all_predictions)
            best_targets = np.array(all_targets)
            best_confusion_matrix = confusion_matrix(best_targets, best_predictions)

            torch.save(model.state_dict(), 'best_fists_model.pth')

        if epoch % 10 == 0:
            print(f'Epoch {epoch}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}')

    # Load best model
    model.load_state_dict(torch.load('best_fists_model.pth'))
    # Display results for best validation performance
    print(f'\n=== BEST VALIDATION PERFORMANCE ===')
    print(f'Best validation accuracy: {best_val_acc:.4f} at epoch {best_epoch}')
    
    if best_confusion_matrix is not None:
        print(f'\nConfusion Matrix at Best Validation Accuracy:')
        print(best_confusion_matrix)
        
        # Calculate metrics
        tn, fp, fn, tp = best_confusion_matrix.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1_score = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
        
        print(f'\nDetailed Metrics at Best Validation:')
        print(f'  Sensitivity (Recall): {sensitivity:.3f}')
        print(f'  Specificity: {specificity:.3f}')
        print(f'  Precision: {precision:.3f}')
        print(f'  F1-Score: {f1_score:.3f}')
        
        # Classification report
        class_names = ['Resting', 'Open/Close Fist']
        print(f'\nClassification Report at Best Validation:')
        print(classification_report(best_targets, best_predictions, 
                                  target_names=class_names, digits=3))

    return train_losses, val_losses, best_confusion_matrix, best_epoch


def train_simple_model_with_universal_best(model, train_loader, val_loader, device, epochs=100, lr=1e-3, 
                                          model_save_dir='models', experiment_name='eeg_classification'):
    """
    Training loop that keeps track of universal best model across multiple runs
    """
    
    # Create model directory if it doesn't exist
    os.makedirs(model_save_dir, exist_ok=True)
    
    # Define file paths
    universal_best_path = os.path.join(model_save_dir, f'{experiment_name}_universal_best.pth')
    current_run_path = os.path.join(model_save_dir, f'{experiment_name}_current_run.pth')
    metadata_path = os.path.join(model_save_dir, f'{experiment_name}_best_metadata.json')
    
    # Load previous best performance if exists
    universal_best_acc = 0
    universal_best_metadata = {}
    
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r') as f:
                universal_best_metadata = json.load(f)
                universal_best_acc = universal_best_metadata.get('best_accuracy', 0)
            print(f"📊 Previous universal best accuracy: {universal_best_acc:.4f}")
            print(f"   From run: {universal_best_metadata.get('run_date', 'Unknown')}")
        except:
            print(" Could not load previous best metadata")

    # Calculate class weights
    train_labels = []
    for _, labels in train_loader:
        train_labels.extend(labels.numpy())

    class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
    pos_weight = torch.tensor(class_weights[1] / class_weights[0]).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    # Current run tracking
    best_val_acc_current_run = 0
    train_losses = []
    val_losses = []
    best_epoch = 0
    best_confusion_matrix = None
    best_predictions = None
    best_targets = None
    
    # Universal best tracking
    found_new_universal_best = False
    
    print(f"\n Starting training...")
    print(f" Target to beat: {universal_best_acc:.4f}")

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
            optimizer.step()

            train_loss += loss.item()

        # Validation
        model.eval()
        correct = 0
        total = 0
        val_loss = 0
        all_predictions = []
        all_targets = []

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

                all_predictions.extend(pred.cpu().numpy().flatten())
                all_targets.extend(target_reshaped.cpu().numpy().flatten())

        val_acc = correct / total
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        scheduler.step(avg_val_loss)

        # Check if this is the best for current run
        if val_acc > best_val_acc_current_run:
            best_val_acc_current_run = val_acc
            best_epoch = epoch

            best_predictions = np.array(all_predictions)
            best_targets = np.array(all_targets)
            best_confusion_matrix = confusion_matrix(best_targets, best_predictions)

            # Always save current run's best
            torch.save(model.state_dict(), current_run_path)
            
            # Check if this beats universal best
            if val_acc > universal_best_acc:
                universal_best_acc = val_acc
                found_new_universal_best = True
                
                # Save new universal best
                torch.save(model.state_dict(), universal_best_path)
                
                # Update metadata
                current_metadata = {
                    'best_accuracy': float(val_acc),
                    'epoch': epoch,
                    'run_date': datetime.now().isoformat(),
                    'train_loss': avg_train_loss,
                    'val_loss': avg_val_loss,
                    'confusion_matrix': best_confusion_matrix.tolist(),
                    'hyperparameters': {
                        'epochs': epochs,
                        'lr': lr,
                        'model_type': type(model).__name__
                    }
                }
                
                with open(metadata_path, 'w') as f:
                    json.dump(current_metadata, f, indent=2)
                
                print(f" NEW UNIVERSAL BEST! Accuracy: {val_acc:.4f} (Previous: {universal_best_metadata.get('best_accuracy', 0):.4f})")

        if epoch % 10 == 0:
            status = "New Best" if val_acc > universal_best_acc else ""
            print(f'Epoch {epoch}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f} {status}')

    # Load best model from current run
    model.load_state_dict(torch.load(current_run_path))
    
    # Print results
    print(f'\n=== CURRENT RUN RESULTS ===')
    print(f'Best validation accuracy this run: {best_val_acc_current_run:.4f} at epoch {best_epoch}')
    
    print(f'\n=== UNIVERSAL BEST COMPARISON ===')
    if found_new_universal_best:
        print(f'🎉 NEW UNIVERSAL BEST ACHIEVED!')
        print(f'   New best: {universal_best_acc:.4f}')
        print(f'   Saved to: {universal_best_path}')
    else:
        print(f'   Current run best: {best_val_acc_current_run:.4f}')
        print(f'   Universal best:   {universal_best_acc:.4f}')
        print(f'   Gap to beat:      {universal_best_acc - best_val_acc_current_run:.4f}')
    
    if best_confusion_matrix is not None:
        print(f'\nCurrent Run - Confusion Matrix at Best Validation:')
        print(best_confusion_matrix)
        
        # Calculate metrics
        tn, fp, fn, tp = best_confusion_matrix.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1_score = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
        
        print(f'\nCurrent Run - Detailed Metrics:')
        print(f'  Sensitivity (Recall): {sensitivity:.3f}')
        print(f'  Specificity: {specificity:.3f}')
        print(f'  Precision: {precision:.3f}')
        print(f'  F1-Score: {f1_score:.3f}')

    return train_losses, val_losses, best_confusion_matrix, best_epoch, found_new_universal_best

def load_universal_best_model(model, model_save_dir='models', experiment_name='eeg_classification'):
    """
    Load the universal best model across all runs
    """
    universal_best_path = os.path.join(model_save_dir, f'{experiment_name}_universal_best.pth')
    metadata_path = os.path.join(model_save_dir, f'{experiment_name}_best_metadata.json')
    
    if not os.path.exists(universal_best_path):
        print(f"No universal best model found at {universal_best_path}")
        return None, None
    
    # Load model
    model.load_state_dict(torch.load(universal_best_path))
    print(f"Loaded universal best model from {universal_best_path}")
    
    # Load metadata if available
    metadata = None
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        print(f"  Universal best performance:")
        print(f"   Accuracy: {metadata['best_accuracy']:.4f}")
        print(f"   From run: {metadata['run_date']}")
        print(f"   Epoch: {metadata['epoch']}")
    
    return model, metadata

def compare_all_runs(model_save_dir='models', experiment_name='eeg_classification'):
    """
    Show performance history across all runs
    """
    metadata_path = os.path.join(model_save_dir, f'{experiment_name}_best_metadata.json')
    
    if not os.path.exists(metadata_path):
        print("No run history found.")
        return
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    print(f"\n=== UNIVERSAL BEST MODEL HISTORY ===")
    print(f"Best Accuracy: {metadata['best_accuracy']:.4f}")
    print(f"Achieved on: {metadata['run_date']}")
    print(f"At epoch: {metadata['epoch']}")
    print(f"Model type: {metadata.get('hyperparameters', {}).get('model_type', 'Unknown')}")
    
    if 'confusion_matrix' in metadata:
        cm = np.array(metadata['confusion_matrix'])
        print(f"Confusion Matrix:")
        print(cm)

def clean_old_models(model_save_dir='models', experiment_name='eeg_classification', keep_universal_best=True):
    """
    Clean up old model files, optionally keeping the universal best
    """
    import glob
    
    pattern = os.path.join(model_save_dir, f'{experiment_name}_*.pth')
    model_files = glob.glob(pattern)
    
    universal_best_path = os.path.join(model_save_dir, f'{experiment_name}_universal_best.pth')
    
    removed_count = 0
    for file_path in model_files:
        if keep_universal_best and file_path == universal_best_path:
            continue
        
        os.remove(file_path)
        removed_count += 1
        print(f"Removed: {file_path}")
    
    print(f"Cleaned {removed_count} old model files")
    if keep_universal_best:
        print(f"Kept universal best: {universal_best_path}")