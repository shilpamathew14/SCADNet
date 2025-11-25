#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
from torch.utils.data import DataLoader

from scadnet.common.seed import set_seeds
from scadnet.stage1.datasets import FundusDataset
from scadnet.stage1.augmentations import SaliencyGuidedAugmentation, saliency_guided_collate_fn
from scadnet.stage1.models import SaliencyAugmentedMoCo
from scadnet.stage1.losses import compute_saliency_divergence, compute_saliency_weighted_loss
from scadnet.stage1.utils import create_train_val_split, to_device_batch, validate_model


# In[ ]:


def train_stage1():

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    print(f"Using device: {device}")
    
    # Set seeds
    seed_value = 1234
    set_seeds(seed_value, use_cuda)
    
    # Clear memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Hyperparameters
    batch_size = 64
    num_epochs = 1400
    learning_rate = 0.0005
    momentum = 0.95
    temperature = 0.07
    queue_size = 65536  
    lambda_sal = 0.5  
    val_ratio = 0.2   
    patience = 500     
    validate_every = 1  
    
    # Data paths
    fundus_dir = #path of  fundus images
    sal_dir = #path of saliency maps
    
   
    full_dataset = FundusDataset(fundus_dir, sal_dir)
    train_dataset, val_dataset = create_train_val_split(full_dataset, val_ratio, seed_value)
    
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=saliency_guided_collate_fn  # Changed this line
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=saliency_guided_collate_fn # Changed this line
    )
    
    print(f"Dataset split - Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Create model

    model = SaliencyAugmentedMoCo(
        base_encoder=efficientnet_b0,
        dim=128,
        queue_size=queue_size,
        momentum=momentum,
        temperature=temperature,
        lambda_sal=lambda_sal
    )
    model = model.to(device)

    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    def save_checkpoint(state, is_best, filename='checkpoint.pth'):
        torch.save(state, filename)
        if is_best:
            save_path = #path to save 
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            shutil.copyfile(filename, save_path)
            print(f"Best model copied to: {save_path}")
    


    best_val_loss = float('inf')
    epochs_without_improvement = 0
    total_training_time = 0
    
    # Track losses for analysis
    train_losses = []
    val_losses = []
    learning_rates = []
    
    for epoch in range(num_epochs):
        model.train()
        epoch_total_loss = 0
        num_batches = 0
        
        start_time = time.time()
        
        # In your training loop, explicitly pass train_mode=True
        for batch_idx, (q_imgs, k_imgs, q_sals, k_sals) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]")):
            # Move to device
            q_imgs = q_imgs.to(device, non_blocking=True)
            k_imgs = k_imgs.to(device, non_blocking=True)
            q_sals = q_sals.to(device, non_blocking=True)
            k_sals = k_sals.to(device, non_blocking=True)
    
           
            total_loss, moco_loss, sal_penalty, logits, labels = model(
                q_imgs, k_imgs, q_sals, k_sals, train_mode=True
                )
    
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
    
            epoch_total_loss += total_loss.item()
            num_batches += 1
        
        # Step learning rate scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        learning_rates.append(current_lr)
        
        # Calculate average training losses
        avg_train_total = epoch_total_loss / num_batches
        train_losses.append(avg_train_total)
        
        end_time = time.time()
        epoch_time = end_time - start_time
        total_training_time += epoch_time
        
        # Validation phase (every N epochs)
        if (epoch + 1) % validate_every == 0:
           
            avg_val_total, avg_val_moco, avg_val_sal = validate_model(model, val_loader, device)
            
            val_losses.append(avg_val_total)
            
            
            is_best = avg_val_total < best_val_loss
            if is_best:
                best_val_loss = avg_val_total
                epochs_without_improvement = 0
                print(f"New best validation loss: {best_val_loss:.4f}")
            else:
                epochs_without_improvement += validate_every
            
            # Print epoch summary with validation
            print(f"\nEpoch [{epoch+1}/{num_epochs}] Summary:")
            print(f"  Train Loss: {avg_train_total:.4f}")
            print(f"  Val Loss: {avg_val_total:.4f}")
            print(f"  LR: {current_lr:.6f}, Time: {epoch_time:.2f}s")
            
            # Save model based on validation loss
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'train_loss': avg_train_total,
                'val_loss': avg_val_total,
                'best_val_loss': best_val_loss,
                'lambda_sal': lambda_sal,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'learning_rates': learning_rates,
                'hyperparameters': {
                    'batch_size': batch_size,
                    'learning_rate': learning_rate,
                    'momentum': momentum,
                    'temperature': temperature,
                    'queue_size': queue_size,
                    'lambda_sal': lambda_sal
                }
            }, is_best=is_best, filename='saliency_moco_checkpoint.pth')
      
            if epochs_without_improvement >= patience:
                print(f"\nEarly stopping triggered! No improvement for {patience} epochs.")
                print(f"Best validation loss: {best_val_loss:.4f}")
                break
                
        else:
            # Print training summary only
            print(f"\nEpoch [{epoch+1}/{num_epochs}] Summary:")
            print(f"  Train Loss: {avg_train_total:.4f}")
            print(f"  LR: {current_lr:.6f}, Time: {epoch_time:.2f}s")

    
    # Print final training summary
    hours, remainder = divmod(total_training_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"\n Training completed!")
    print(f"Total training time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
    print(f"Best validation loss achieved: {best_val_loss:.4f}")
    

    # Plot training curves
    if val_losses:
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Training Loss')
   
        val_epochs = [i * validate_every for i in range(len(val_losses))]
        plt.plot(val_epochs, val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(val_epochs, val_losses, label='Validation Loss', color='orange')
        plt.axhline(y=best_val_loss, color='r', linestyle='--', label=f'Best Val Loss: {best_val_loss:.4f}')
        plt.xlabel('Epoch')
        plt.ylabel('Validation Loss')
        plt.title('Validation Loss Progress')
        plt.legend()
        plt.grid(True)
        
        
        plt.tight_layout()
        plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
        plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




