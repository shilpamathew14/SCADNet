#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn.functional as F

def compute_saliency_divergence(self, sal_q, sal_k, method='cosine'):
        
        # Downsample and flatten saliency maps for efficiency
        sal_q_down = F.adaptive_avg_pool2d(sal_q, (64, 64))
        sal_k_down = F.adaptive_avg_pool2d(sal_k, (64, 64))
        
        # Convert to grayscale if needed and flatten
        if sal_q_down.shape[1] == 3:
            sal_q_flat = torch.mean(sal_q_down, dim=1, keepdim=True).flatten(1)
            sal_k_flat = torch.mean(sal_k_down, dim=1, keepdim=True).flatten(1)
        else:
            sal_q_flat = sal_q_down.flatten(1)
            sal_k_flat = sal_k_down.flatten(1)
        
        # Normalize to probability distributions
        sal_q_norm = F.softmax(sal_q_flat, dim=1)
        sal_k_norm = F.softmax(sal_k_flat, dim=1)
        
        if method == 'cosine':
            cosine_sim = F.cosine_similarity(sal_q_norm, sal_k_norm, dim=1)
            divergence = 1 - cosine_sim  # Convert to distance
        elif method == 'kl':
            kl1 = F.kl_div(torch.log(sal_q_norm + 1e-8), sal_k_norm, reduction='none').sum(dim=1)
            kl2 = F.kl_div(torch.log(sal_k_norm + 1e-8), sal_q_norm, reduction='none').sum(dim=1)
            divergence = 0.5 * (kl1 + kl2)
        else:  # 'mse'
            divergence = F.mse_loss(sal_q_norm, sal_k_norm, reduction='none').mean(dim=1)
            
        return divergence

def compute_saliency_weighted_loss(self, logits, labels, sal_q, sal_k, sal_neg_queue):
       
        batch_size = logits.shape[0]
        
        # Standard MoCo loss
        moco_loss = F.cross_entropy(logits, labels)
        
        # Compute saliency divergence for positive pairs
        sal_div_pos = self.compute_saliency_divergence(sal_q, sal_k, method='cosine')
        
        
        target_divergence = 0.3  
        saliency_penalty = torch.abs(sal_div_pos - target_divergence)
        
        # Final loss
        total_loss = moco_loss + self.lambda_sal * saliency_penalty.mean()
        
        return total_loss, moco_loss, saliency_penalty.mean()

