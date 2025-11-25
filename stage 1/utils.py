#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
from torch.utils.data import DataLoader, random_split



# In[ ]:


def to_device_batch(batch, device):
    (im_q_fg, im_q_bg, im_k_fg, im_k_bg, sal_q, sal_k) = batch
    im_q_fg = to_device_scales(im_q_fg, device)
    im_q_bg = to_device_scales(im_q_bg, device)
    im_k_fg = to_device_scales(im_k_fg, device)
    im_k_bg = to_device_scales(im_k_bg, device)
    sal_q   = to_device_scales(sal_q,   device)
    sal_k   = to_device_scales(sal_k,   device)
    return im_q_fg, im_q_bg, im_k_fg, im_k_bg, sal_q, sal_k


# In[ ]:


def create_train_val_split(dataset, val_ratio=0.2, seed=1234):
    
    total_size = len(dataset)
    val_size = int(val_ratio * total_size)
    train_size = total_size - val_size
    
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
    
    return train_dataset, val_dataset

def validate_model(model, val_loader, device):
    
    model.eval()
    total_val_loss = 0
    total_moco_loss = 0
    total_sal_penalty = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch_idx, (q_imgs, k_imgs, q_sals, k_sals) in enumerate(val_loader):
            # Move to device
            q_imgs = q_imgs.to(device, non_blocking=True)
            k_imgs = k_imgs.to(device, non_blocking=True)
            q_sals = q_sals.to(device, non_blocking=True)
            k_sals = k_sals.to(device, non_blocking=True)

           
            total_loss, moco_loss, sal_penalty, logits, labels = model(
                q_imgs, k_imgs, q_sals, k_sals, train_mode=False
            )
            
            
            total_val_loss += total_loss.item()
            total_moco_loss += moco_loss.item()
            total_sal_penalty += sal_penalty.item()
            num_batches += 1
    
    avg_val_loss = total_val_loss / num_batches if num_batches > 0 else float('inf')
    avg_val_moco = total_moco_loss / num_batches if num_batches > 0 else float('inf')
    avg_val_sal = total_sal_penalty / num_batches if num_batches > 0 else float('inf')
    
    return avg_val_loss, avg_val_moco, avg_val_sal


# In[ ]:


def load_checkpoint(model, optimizer, checkpoint_path):
 
    try:
        print(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
 
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        resume_info = {
            'start_epoch': checkpoint['epoch'],
            'best_loss': checkpoint.get('best_loss', checkpoint['total_loss']),
            'lambda_sal': checkpoint['lambda_sal'],
            'scale_sizes': checkpoint['scale_sizes'],
            'queue_size': checkpoint['queue_size'],
            'last_total_loss': checkpoint['total_loss'],
            'last_moco_loss': checkpoint['moco_loss'],
            'last_sal_penalty': checkpoint['saliency_penalty']
        }
        
        print(f"Checkpoint loaded successfully!")
        print(f" Resume from epoch: {resume_info['start_epoch']}")
        print(f" Last total loss: {resume_info['last_total_loss']:.4f}")
        print(f" Best loss so far: {resume_info['best_loss']:.4f}")

        return resume_info



def find_latest_checkpoint(checkpoint_dir='checkpoints'):
 
    latest_path = os.path.join(checkpoint_dir, 'latest_checkpoint.pth')
    if os.path.exists(latest_path):
        return latest_path
    return None

