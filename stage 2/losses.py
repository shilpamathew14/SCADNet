#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F


# In[ ]:


class DicewithCE(nn.Module):
    """
    Combination of Cross Entropy and Dice Loss.
    """
    
    def __init__(self, alpha=0.4):
        super().__init__()
        self.alpha = alpha
        # Weighted CE for rare classes
        self.ce = nn.CrossEntropyLoss(weight=torch.FloatTensor([1.0, 3.0, 2.0, 5.0, 5.0]))
        self.dice = self.balanced_dice_loss
    
    def balanced_dice_loss(self, inputs, targets):
        inputs = F.softmax(inputs, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=inputs.shape[1]).permute(0, 3, 1, 2).float()
        
        dice_scores = []
        # Calculate Dice for each lesion class (skip background)
        for i in range(1, inputs.shape[1]):
            # FIX: Use .contiguous().view() or .reshape()
            input_flat = inputs[:, i].contiguous().view(-1)
            target_flat = targets_one_hot[:, i].contiguous().view(-1)
            
            if target_flat.sum() > 0:  # Only if positive samples exist
                intersection = (input_flat * target_flat).sum()
                dice = (2. * intersection + 1e-6) / (input_flat.sum() + target_flat.sum() + 1e-6)
                dice_scores.append(dice)
        
        if len(dice_scores) == 0:
            return torch.tensor(0.0, requires_grad=True, device=inputs.device)
        
        return 1 - torch.stack(dice_scores).mean()
    
    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        dice_loss = self.balanced_dice_loss(inputs, targets)
        return self.alpha * ce_loss + (1 - self.alpha) * dice_loss


# In[ ]:


def calculate_class_weights(dataset_loader, num_classes=5):
  
    class_counts = torch.zeros(num_classes)
    total_pixels = 0
    
    
    for batch_idx, (_, masks) in enumerate(dataset_loader):
        for class_id in range(num_classes):
            class_counts[class_id] += (masks == class_id).sum().item()
        total_pixels += masks.numel()
        
        if (batch_idx + 1) % 100 == 0:
            print(f"Processed {batch_idx + 1} batches...")
    
    # Calculate weights (inverse frequency)
    class_frequencies = class_counts / total_pixels
    class_weights = 1.0 / (class_frequencies + 1e-6)
    
    # Normalize weights
    class_weights = class_weights / class_weights.sum() * num_classes
    
    print(f"Class frequencies: {class_frequencies}")
    print(f"Class weights: {class_weights}")
    
    return class_weights.tolist()


# In[ ]:


def get_loss_function(loss_type="weighted_ce", class_weights=None):
 
    if loss_type == "ce":
        return nn.CrossEntropyLoss()
    
    elif loss_type == "dicewithCE":
        return DicewithCE(alpha=0.4)

    else:
        raise ValueError(f"Unknown loss type: {loss_type}")



# In[ ]:




