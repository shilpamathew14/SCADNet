#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F


# In[ ]:


class SaliencyAugmentedMoCo(nn.Module):
    def __init__(self, base_encoder, dim=128, queue_size=65536, momentum=0.99, temperature=0.07, lambda_sal=0.5):
        super(SaliencyAugmentedMoCo, self).__init__()

        self.queue_size = queue_size
        self.m = momentum
        self.T = temperature
        self.lambda_sal = lambda_sal

        # Encoders
        self.encoder_q = base_encoder(num_classes=dim)
        self.encoder_k = base_encoder(num_classes=dim)

        # Initialize key encoder with query encoder weights
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        # Feature queue
        self.register_buffer("queue", torch.randn(dim, queue_size))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        # Saliency queue for enhanced loss computation
        self.register_buffer("saliency_queue", torch.zeros(queue_size, 1, 64, 64))
        
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

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """Momentum update of the key encoder"""
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, saliency_k):
        """Update the queue with new keys and saliency maps"""
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)

        # Prepare saliency maps for queue storage
        sal_k_down = F.adaptive_avg_pool2d(saliency_k, (64, 64))
        if sal_k_down.shape[1] == 3:
            sal_k_down = torch.mean(sal_k_down, dim=1, keepdim=True)

        # Handle queue update with proper boundary checks
        if ptr + batch_size <= self.queue_size:
            self.queue[:, ptr:ptr + batch_size] = keys.T
            self.saliency_queue[ptr:ptr + batch_size] = sal_k_down
        else:
            # Handle wraparound
            remaining = self.queue_size - ptr
            self.queue[:, ptr:] = keys[:remaining].T
            self.queue[:, :batch_size - remaining] = keys[remaining:].T
            
            self.saliency_queue[ptr:] = sal_k_down[:remaining]
            self.saliency_queue[:batch_size - remaining] = sal_k_down[remaining:]

        ptr = (ptr + batch_size) % self.queue_size
        self.queue_ptr[0] = ptr

    def forward(self, im_q, im_k, saliency_q, saliency_k, train_mode=True):
        
        # Compute features
        q = F.normalize(self.encoder_q(im_q), dim=1)  # FG features
        
        with torch.no_grad():
            if train_mode:
                self._momentum_update_key_encoder()
            k = F.normalize(self.encoder_k(im_k), dim=1)  # BG features

        # Compute logits - this creates FG vs BG contrast!
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)  # [B, 1]
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])  # [B, K]
        logits = torch.cat([l_pos, l_neg], dim=1)  # [B, 1+K]
        logits /= self.T
        
        # Standard labels (positive pairs are at index 0)
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        
        # Compute enhanced saliency loss
        total_loss, moco_loss, sal_penalty = self.compute_saliency_weighted_loss(
            logits, labels, saliency_q, saliency_k, self.saliency_queue
        )
        
        # Update queue with key features and saliency
        if train_mode:
            self._dequeue_and_enqueue(k, saliency_k)
        
        return total_loss, moco_loss, sal_penalty, logits, labels


# In[ ]:


class SaliencyAugmentedInfoNCE(nn.Module):
    def __init__(self):
        super(SaliencyAugmentedInfoNCE, self).__init__()

    def forward(self, total_loss, moco_loss, sal_penalty):
      
        return {
            'total_loss': total_loss,
            'moco_loss': moco_loss, 
            'saliency_penalty': sal_penalty
        }

