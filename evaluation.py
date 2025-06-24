import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, classification_report

def evaluate_simple_model(model, test_loader, device):
    """Evaluate the model"""
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            probs = torch.sigmoid(outputs).cpu().numpy()
            preds = (probs > 0.5).astype(int)

            all_preds.extend(preds.flatten())
            all_probs.extend(probs.flatten())
            all_labels.extend(labels.cpu().numpy())

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)

def plot_results(train_losses, val_losses, labels, preds, probs, save_path='fists_result.png'):
    """Plot training curves, confusion matrix, and ROC curve"""
    name = save_path.split('_')[0]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Training curves
    axes[0].plot(train_losses, label='Train Loss')
    axes[0].plot(val_losses, label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].set_title(f'Training Curves {name}')

    # Confusion Matrix
    cm = confusion_matrix(labels, preds)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[1],
                xticklabels=["Resting", "Open/Close Fist"],
                yticklabels=["Resting", "Open/Close Fist"])
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")
    axes[1].set_title(f"Confusion Matrix {name}")

    # ROC Curve
    fpr, tpr, _ = roc_curve(labels, probs)
    auc_score = roc_auc_score(labels, probs)
    axes[2].plot(fpr, tpr, label=f"AUC = {auc_score:.3f}")
    axes[2].plot([0, 1], [0, 1], linestyle="--", color="gray")
    axes[2].set_xlabel("False Positive Rate")
    axes[2].set_ylabel("True Positive Rate")
    axes[2].set_title(f"ROC Curve {name}")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

    return auc_score

def plot_results_short(labels, preds, probs, save_path='fists_result.png'):
    """Plot confusion matrix, and ROC curve"""
    name = save_path.split('_')[0]
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Confusion Matrix
    cm = confusion_matrix(labels, preds)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
                xticklabels=["Resting", f"Open/Close {name}"],
                yticklabels=["Resting", f"Open/Close {name}"])
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")
    axes[0].set_title(f"Confusion Matrix {name}")

    # ROC Curve
    fpr, tpr, _ = roc_curve(labels, probs)
    auc_score = roc_auc_score(labels, probs)
    axes[1].plot(fpr, tpr, label=f"AUC = {auc_score:.3f}")
    axes[1].plot([0, 1], [0, 1], linestyle="--", color="gray")
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].set_title(f"ROC Curve {name}")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

    return auc_score

def print_classification_results(labels, preds, probs, name="fists"):
    """Print classification report and AUC score"""
    print("\nClassification Report:")
    print(classification_report(labels, preds, target_names=["Resting", f"Open/Close {name}"]))
    
    auc_score = roc_auc_score(labels, probs)
    print(f"\nAUC Score: {auc_score:.3f}")
    
    return auc_score