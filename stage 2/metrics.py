#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import numpy as np


# In[ ]:


def validate_model_simple(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    class_correct = torch.zeros(5)
    class_total = torch.zeros(5)
    
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            val_loss += loss.item()
            
            # Calculate per-class accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += masks.numel()
            correct += (predicted == masks).sum().item()
            
            # Per-class statistics
            for c in range(5):
                class_mask = (masks == c)
                if class_mask.sum() > 0:
                    class_correct[c] += ((predicted == c) & (masks == c)).sum().item()
                    class_total[c] += class_mask.sum().item()
    
    avg_loss = val_loss / len(val_loader)
    accuracy = correct / total
    
    # Print per-class accuracies for debugging
    print(f"    Per-class accuracies:")
    class_names = ['BG', 'MA', 'HE', 'SE', 'HM']
    for i, name in enumerate(class_names):
        if class_total[i] > 0:
            class_acc = class_correct[i] / class_total[i]
            print(f"    {name}: {class_acc:.3f}")
    
    return avg_loss, accuracy


# In[ ]:


def calculate_per_lesion_metrics(model, val_loader, device, num_classes=5):
    """
    Calculate PR-AUC, Dice, and IoU for each lesion type.
    Classes: 0=background, 1=MA, 2=EX, 3=SE, 4=HE
    """
    model.eval()
    
    lesion_names = ['MA', 'EX', 'SE', 'HE']
    
    # Storage for metrics
    all_probs = {lesion: [] for lesion in lesion_names}
    all_masks = {lesion: [] for lesion in lesion_names}
    dice_scores = {lesion: [] for lesion in lesion_names}
    iou_scores = {lesion: [] for lesion in lesion_names}
    
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            
            probs = probs.cpu().numpy()
            preds = preds.cpu().numpy()
            masks = masks.cpu().numpy()
            
            # Calculate for each lesion type (class 1-4)
            for class_idx, lesion in enumerate(lesion_names, start=1):
                for i in range(len(masks)):
                    pred_binary = (preds[i] == class_idx).astype(np.uint8)
                    mask_binary = (masks[i] == class_idx).astype(np.uint8)
                    prob_map = probs[i, class_idx]
                    
                    # Store for PR-AUC
                    all_probs[lesion].append(prob_map.flatten())
                    all_masks[lesion].append(mask_binary.flatten())
                    
                    # Calculate Dice and IoU
                    tp = np.sum((pred_binary == 1) & (mask_binary == 1))
                    fp = np.sum((pred_binary == 1) & (mask_binary == 0))
                    fn = np.sum((pred_binary == 0) & (mask_binary == 1))
                    
                    dice = (2 * tp) / (2 * tp + fp + fn + 1e-8)
                    iou = tp / (tp + fp + fn + 1e-8)
                    
                    dice_scores[lesion].append(dice)
                    iou_scores[lesion].append(iou)
    
    # Calculate PR-AUC for each lesion
    results = {}
    for lesion in lesion_names:
        probs_concat = np.concatenate(all_probs[lesion])
        masks_concat = np.concatenate(all_masks[lesion])
        
        precision, recall, _ = precision_recall_curve(masks_concat, probs_concat)
        pr_auc = auc(recall, precision) * 100  # Convert to percentage
        
        results[lesion] = {
            'pr_auc': pr_auc,
            'dice': np.mean(dice_scores[lesion]) * 100,
            'iou': np.mean(iou_scores[lesion]) * 100
        }
    
    # Calculate mean across all lesions
    results['Mean'] = {
        'pr_auc': np.mean([results[l]['pr_auc'] for l in lesion_names]),
        'dice': np.mean([results[l]['dice'] for l in lesion_names]),
        'iou': np.mean([results[l]['iou'] for l in lesion_names])
    }
    
    return results



# In[ ]:


def plot_kfold_loss_overlayed(results_path, epochs_per_fold=40):
    """
    Plot all folds on same graph with x-axis 0-40 epochs (overlayed).
    Each fold shows its own 0-40 epoch progression.
    """
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    n_folds = results['n_folds']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = plt.cm.tab10(np.linspace(0, 1, n_folds))
    
    # Plot training and validation loss
    for i, fold in enumerate(results['folds']):
        train_losses = fold['train_losses']
        val_losses = fold['val_losses']
        
        # Each fold starts from epoch 0
        train_epochs = np.arange(0, len(train_losses) * 5, 5)  # Assuming stored every 5 epochs
        val_epochs = np.arange(0, len(val_losses) * 5, 5)
        
        ax1.plot(train_epochs, train_losses, '-', color=colors[i], 
                 label=f'Fold {i+1} Train', alpha=0.6, linewidth=1.5)
        ax1.plot(val_epochs, val_losses, '--', color=colors[i], 
                 label=f'Fold {i+1} Val', alpha=0.8, linewidth=2, marker='o', markersize=4)
    
    ax1.set_xlabel('Epochs (per fold)', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title(f'Training and Validation Loss ({n_folds} Folds Overlayed)', fontsize=13)
    ax1.legend(fontsize=8, ncol=2, loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, epochs_per_fold)
    
    # Plot validation accuracy
    for i, fold in enumerate(results['folds']):
        val_accuracies = fold['val_accuracies']
        val_epochs = np.arange(0, len(val_accuracies) * 5, 5)
        
        ax2.plot(val_epochs, val_accuracies, '-', color=colors[i], 
                 label=f'Fold {i+1}', alpha=0.8, linewidth=2, marker='o', markersize=4)
    
    ax2.set_xlabel('Epochs (per fold)', fontsize=12)
    ax2.set_ylabel('Validation Accuracy', fontsize=12)
    ax2.set_title(f'Validation Accuracy ({n_folds} Folds Overlayed)', fontsize=13)
    ax2.legend(fontsize=9, loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, epochs_per_fold)
    
    plt.tight_layout()
    plt.savefig(results_path.replace('.json', '_overlayed_loss.png'), dpi=300, bbox_inches='tight')
    plt.show()




def generate_detailed_fold_report(results_path):
    """Generate per-fold, per-lesion results table."""
    
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    lesion_names = ['MA', 'EX', 'SE', 'HE', 'Mean']
    
    # PR-AUC
    print("\nPR-AUC:")
    print(f"{'Fold':<8} {'MA':<12} {'EX':<12} {'SE':<12} {'HE':<12} {'Mean':<12}")
   
    
    pr_auc_data = []
    for fold_idx, fold_data in enumerate(results['folds']):
        metrics = fold_data['lesion_metrics']
        row = [fold_idx + 1]
        for lesion in lesion_names:
            row.append(metrics[lesion]['pr_auc'])
        pr_auc_data.append(row)
        print(f"{row[0]:<8} {row[1]:<12.2f} {row[2]:<12.2f} {row[3]:<12.2f} {row[4]:<12.2f} {row[5]:<12.2f}")
    
    pr_auc_array = np.array(pr_auc_data)
    means = pr_auc_array[:, 1:].mean(axis=0)
    stds = pr_auc_array[:, 1:].std(axis=0)
    print(f"{'Mean':<8} {means[0]:<12.2f} {means[1]:<12.2f} {means[2]:<12.2f} {means[3]:<12.2f} {means[4]:<12.2f}")
    print(f"{'Std':<8} {stds[0]:<12.2f} {stds[1]:<12.2f} {stds[2]:<12.2f} {stds[3]:<12.2f} {stds[4]:<12.2f}")
    
    # Dice
    print("\nDice Score:")
    print(f"{'Fold':<8} {'MA':<12} {'EX':<12} {'SE':<12} {'HE':<12} {'Mean':<12}")

    
    dice_data = []
    for fold_idx, fold_data in enumerate(results['folds']):
        metrics = fold_data['lesion_metrics']
        row = [fold_idx + 1]
        for lesion in lesion_names:
            row.append(metrics[lesion]['dice'])
        dice_data.append(row)
        print(f"{row[0]:<8} {row[1]:<12.2f} {row[2]:<12.2f} {row[3]:<12.2f} {row[4]:<12.2f} {row[5]:<12.2f}")
    
    dice_array = np.array(dice_data)
    means = dice_array[:, 1:].mean(axis=0)
    stds = dice_array[:, 1:].std(axis=0)
    print(f"{'Mean':<8} {means[0]:<12.2f} {means[1]:<12.2f} {means[2]:<12.2f} {means[3]:<12.2f} {means[4]:<12.2f}")
    print(f"{'Std':<8} {stds[0]:<12.2f} {stds[1]:<12.2f} {stds[2]:<12.2f} {stds[3]:<12.2f} {stds[4]:<12.2f}")
    
    # IoU
    print("\nIoU:")
    print(f"{'Fold':<8} {'MA':<12} {'EX':<12} {'SE':<12} {'HE':<12} {'Mean':<12}")
    
    iou_data = []
    for fold_idx, fold_data in enumerate(results['folds']):
        metrics = fold_data['lesion_metrics']
        row = [fold_idx + 1]
        for lesion in lesion_names:
            row.append(metrics[lesion]['iou'])
        iou_data.append(row)
        print(f"{row[0]:<8} {row[1]:<12.2f} {row[2]:<12.2f} {row[3]:<12.2f} {row[4]:<12.2f} {row[5]:<12.2f}")
    
    iou_array = np.array(iou_data)
    means = iou_array[:, 1:].mean(axis=0)
    stds = iou_array[:, 1:].std(axis=0)
    print(f"{'Mean':<8} {means[0]:<12.2f} {means[1]:<12.2f} {means[2]:<12.2f} {means[3]:<12.2f} {means[4]:<12.2f}")
    print(f"{'Std':<8} {stds[0]:<12.2f} {stds[1]:<12.2f} {stds[2]:<12.2f} {stds[3]:<12.2f} {stds[4]:<12.2f}")

    
    # Save to CSV
    pd.DataFrame(pr_auc_data, columns=['Fold', 'MA', 'EX', 'SE', 'HE', 'Mean']).to_csv(
        results_path.replace('.json', '_pr_auc.csv'), index=False)
    pd.DataFrame(dice_data, columns=['Fold', 'MA', 'EX', 'SE', 'HE', 'Mean']).to_csv(
        results_path.replace('.json', '_dice.csv'), index=False)
    pd.DataFrame(iou_data, columns=['Fold', 'MA', 'EX', 'SE', 'HE', 'Mean']).to_csv(
        results_path.replace('.json', '_iou.csv'), index=False)

    return pr_auc_array, dice_array, iou_array





# In[ ]:





# In[ ]:




