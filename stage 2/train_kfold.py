#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold

from scadnet.common.seed import set_seeds
from scadnet.stage2.datasets import IDRIDDirectPatchDataset
from scadnet.stage2.model import create_model
from scadnet.stage2.losses import get_loss_function
from scadnet.stage2.metrics import validate_model_simple, calculate_per_lesion_metrics


# In[ ]:


def train_kfold_cross_validation(model_class, n_folds=5, total_epochs=100, resume=False):

    
    image_patches_dir =#path of images
    mask_patches_dir = #path of masks
    save_path = #save path 
    
    os.makedirs(save_path, exist_ok=True)
    
    # Training parameters
    batch_size = 32
    initial_lr = 0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Dataset preparation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    full_dataset = IDRIDDirectPatchDataset(
        image_patches_dir=image_patches_dir,
        mask_patches_dir=mask_patches_dir,
        transform=transform
    )
    
    # K-Fold setup
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # Results storage
    checkpoint_path = os.path.join(save_path, 'kfold_checkpoint.pth')
    results_path = os.path.join(save_path, 'kfold_results.json')
    
    # Resume from checkpoint if requested
    start_fold = 0
    fold_results = {
        'n_folds': n_folds,
        'total_epochs': total_epochs,
        'folds': []
    }
    
    if resume and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        start_fold = checkpoint['current_fold']
        fold_results = checkpoint['fold_results']
        print(f"Resuming from Fold {start_fold + 1}/{n_folds}")
        print(f"Completed folds: {start_fold}")
    else:
        fold_results = {
            'n_folds': n_folds,
            'total_epochs': total_epochs,
            'folds': []
        }
    
    print(f"K-FOLD CROSS VALIDATION ({n_folds} folds)")

    # Loop through each fold
    for fold, (train_ids, val_ids) in enumerate(kfold.split(range(len(full_dataset)))):
        # Skip completed folds
        if fold < start_fold:
            continue
        
        print(f"FOLD {fold + 1}/{n_folds}")

        
        # Create fresh model for this fold
        model = model_class()
        model = model.to(device)
        
        # Create data loaders for this fold
        train_subsampler = Subset(full_dataset, train_ids)
        val_subsampler = Subset(full_dataset, val_ids)
        
        train_loader = DataLoader(train_subsampler, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subsampler, batch_size=batch_size, shuffle=False)
        
        # Loss and optimizer
        criterion = get_loss_function("dicewithCE").to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)
        
        # Training tracking
        best_val_loss = float('inf')
        start_epoch = 0
        fold_log = {
            'fold': fold + 1,
            'train_losses': [],
            'val_losses': [],
            'val_accuracies': [],
            'best_val_loss': None,
            'best_epoch': None
        }
        
        # Resume from fold checkpoint if exists
        fold_checkpoint_path = os.path.join(save_path, f'fold_{fold+1}_latest.pth')
        if resume and os.path.exists(fold_checkpoint_path):
            fold_checkpoint = torch.load(fold_checkpoint_path)
            model.load_state_dict(fold_checkpoint['model_state_dict'])
            optimizer.load_state_dict(fold_checkpoint['optimizer_state_dict'])
            start_epoch = fold_checkpoint['epoch'] + 1
            best_val_loss = fold_checkpoint['best_val_loss']
            fold_log = fold_checkpoint['fold_log']
            print(f"Resuming Fold {fold + 1} from epoch {start_epoch + 1}")
        
        # Training loop for this fold
        # Training loop for this fold
        for epoch in range(start_epoch, total_epochs):
            # Training phase
            model.train()
            train_loss = 0
            for images, masks in train_loader:
                images, masks = images.to(device), masks.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
    
            avg_train_loss = train_loss / len(train_loader)
    
            # Validation phase (every 5 epochs OR final epoch)
            if epoch % 5 == 0 or epoch == total_epochs - 1:
                val_loss, val_accuracy = validate_model_simple(model, val_loader, criterion, device)
        
                fold_log['train_losses'].append(avg_train_loss)
                fold_log['val_losses'].append(val_loss)
                fold_log['val_accuracies'].append(val_accuracy)
        
                # Check if best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    fold_log['best_val_loss'] = val_loss
                    fold_log['best_epoch'] = epoch + 1
        
                # At final epoch, calculate lesion metrics
                if epoch == total_epochs - 1:
                    lesion_metrics = calculate_per_lesion_metrics(model, val_loader, device)
                    fold_log['lesion_metrics'] = lesion_metrics
            
                    # Save final model with lesion metrics
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'fold': fold + 1,
                        'val_loss': val_loss,
                        'val_accuracy': val_accuracy,
                        'lesion_metrics': lesion_metrics
                    }, os.path.join(save_path, f'fold_{fold+1}_final.pth'))
            
                    print(f"\nFold {fold+1} Final Results:")
                    print(f"  MA  - Dice: {lesion_metrics['MA']['dice']:.2f}, IoU: {lesion_metrics['MA']['iou']:.2f}, PR-AUC: {lesion_metrics['MA']['pr_auc']:.2f}")
                    print(f"  EX  - Dice: {lesion_metrics['EX']['dice']:.2f}, IoU: {lesion_metrics['EX']['iou']:.2f}, PR-AUC: {lesion_metrics['EX']['pr_auc']:.2f}")
                    print(f"  SE  - Dice: {lesion_metrics['SE']['dice']:.2f}, IoU: {lesion_metrics['SE']['iou']:.2f}, PR-AUC: {lesion_metrics['SE']['pr_auc']:.2f}")
                    print(f"  HE  - Dice: {lesion_metrics['HE']['dice']:.2f}, IoU: {lesion_metrics['HE']['iou']:.2f}, PR-AUC: {lesion_metrics['HE']['pr_auc']:.2f}")
                    print(f"  Mean- Dice: {lesion_metrics['Mean']['dice']:.2f}, IoU: {lesion_metrics['Mean']['iou']:.2f}, PR-AUC: {lesion_metrics['Mean']['pr_auc']:.2f}")
                else:
                    print(f"Epoch {epoch+1}/{total_epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.3f}")
        
                # Save checkpoint every 10 epochs or at final epoch
                if epoch % 10 == 0 or epoch == total_epochs - 1:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'best_val_loss': best_val_loss,
                        'fold_log': fold_log
                    }, os.path.join(save_path, f'fold_{fold+1}_latest.pth'))
    
            # Update learning rate
            lr = step_decay(epoch, initial_lr=initial_lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        
                # Store fold results
        fold_results['folds'].append(fold_log)
        
        # Save global checkpoint after completing each fold
        torch.save({
            'current_fold': fold + 1,  # Next fold to train
            'fold_results': fold_results,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }, checkpoint_path)
        
        # Save intermediate results
        with open(results_path, 'w') as f:
            json.dump(fold_results, f, indent=2)
        
        print(f"\nFold {fold + 1} Summary:")
        print(f"  Best Val Loss: {fold_log['best_val_loss']:.4f} at epoch {fold_log['best_epoch']}")
        print(f"  Final Val Accuracy: {fold_log['val_accuracies'][-1]:.3f}")
        print(f"  Checkpoint saved: fold_{fold+1}_latest.pth")
    
    # Calculate and report aggregate results
    all_best_losses = [f['best_val_loss'] for f in fold_results['folds']]
    all_final_accs = [f['val_accuracies'][-1] for f in fold_results['folds']]
    
    fold_results['summary'] = {
        'mean_best_val_loss': np.mean(all_best_losses),
        'std_best_val_loss': np.std(all_best_losses),
        'mean_final_accuracy': np.mean(all_final_accs),
        'std_final_accuracy': np.std(all_final_accs),
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Save complete results
    results_path = os.path.join(save_path, 'kfold_results.json')
    with open(results_path, 'w') as f:
        json.dump(fold_results, f, indent=2)
    
    # Delete global checkpoint after completion
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print(f"\nGlobal checkpoint deleted (training completed)")
    
    # Print final summary

    print(f"CROSS VALIDATION SUMMARY")
    print(f"Mean Best Val Loss: {fold_results['summary']['mean_best_val_loss']:.4f} "
          f"± {fold_results['summary']['std_best_val_loss']:.4f}")
    print(f"Mean Final Accuracy: {fold_results['summary']['mean_final_accuracy']:.3f} "
          f"± {fold_results['summary']['std_final_accuracy']:.3f}")
    print(f"\nPer-fold results:")
    for i, fold_log in enumerate(fold_results['folds']):
        print(f"  Fold {i+1}: Best Loss = {fold_log['best_val_loss']:.4f}, "
              f"Final Acc = {fold_log['val_accuracies'][-1]:.3f}")
    print(f"\nResults saved to: {results_path}")
    
    return fold_results


# In[ ]:





# In[ ]:





# In[ ]:




