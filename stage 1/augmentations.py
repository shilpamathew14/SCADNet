#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch


def create_saliency_masks(saliency_tensor, foreground_threshold=0.5, background_threshold=0.4):
    """
    Create foreground and background masks from saliency maps with conservative thresholds
    Args:
        saliency_tensor: Normalized saliency map [0, 1]
        foreground_threshold: Threshold for high-saliency regions (lowered)
        background_threshold: Threshold for low-saliency regions (increased)
    Returns:
        foreground_mask, background_mask (binary tensors)
    """
    # Normalize saliency to [0, 1] if not already
    sal_norm = (saliency_tensor - saliency_tensor.min()) / (saliency_tensor.max() - saliency_tensor.min() + 1e-8)
    
    # Apply slight Gaussian blur to smooth saliency map
    sal_smooth = F.avg_pool2d(sal_norm.unsqueeze(0), kernel_size=3, stride=1, padding=1).squeeze(0)
    
    # Create masks with overlap region (pixels between thresholds remain unchanged)
    foreground_mask = (sal_smooth >= foreground_threshold).float()
    background_mask = (sal_smooth <= background_threshold).float()
    
    return foreground_mask, background_mask

def apply_saliency_masking(image_tensor, saliency_tensor, mask_type='foreground', 
                          noise_type='gaussian', noise_std=0.05, smooth_edges=True):
    
    # Ensure saliency is single channel
    if saliency_tensor.shape[0] == 3:
        saliency_tensor = torch.mean(saliency_tensor, dim=0, keepdim=True)
    
    # Create masks with more conservative thresholds
    fg_mask, bg_mask = create_saliency_masks(saliency_tensor, 
                                           foreground_threshold=0.5,  # Lowered
                                           background_threshold=0.4)   # Increased
    
    # Select appropriate mask
    if mask_type == 'foreground':
        keep_mask = fg_mask
    else:  # background
        keep_mask = bg_mask
    
    # Smooth mask edges to reduce artifacts
    if smooth_edges:
        # Apply Gaussian blur to soften mask edges
        keep_mask = F.avg_pool2d(keep_mask.unsqueeze(0), kernel_size=5, stride=1, padding=2).squeeze(0)
    
    # Expand mask to match image channels
    keep_mask = keep_mask.expand_as(image_tensor)
    
    # Create replacement values based on noise type
    if noise_type == 'gaussian':
        # Reduced noise and blend with original
        img_mean = torch.mean(image_tensor, dim=[1, 2], keepdim=True)
        noise = torch.randn_like(image_tensor) * noise_std + img_mean
        # Blend noise with original image for smoother transition
        replacement = 0.7 * img_mean.expand_as(image_tensor) + 0.3 * noise
    elif noise_type == 'mean':
        # Use image mean
        img_mean = torch.mean(image_tensor, dim=[1, 2], keepdim=True)
        replacement = img_mean.expand_as(image_tensor)
    else:  # 'zero'
        replacement = torch.zeros_like(image_tensor)
    
    # Apply masking with smoother blending
    # Use weighted combination instead of hard masking
    mask_strength = 0.8  # Reduce masking strength
    masked_image = image_tensor * (keep_mask * mask_strength + (1 - mask_strength)) + \
                   replacement * (1 - keep_mask) * mask_strength
    
    return masked_image


# In[ ]:


class SaliencyGuidedAugmentation:
    """
    Single-scale saliency-guided augmentation for creating foreground/background views
    """
    def __init__(self, mean=None, std=None, image_size=256,
                 fg_threshold=0.6, bg_threshold=0.3, noise_type='gaussian', noise_std=0.05):
        """
        Args with more conservative defaults
        """
        self.image_size = image_size
        self.mean = mean or [0.485, 0.456, 0.406]
        self.std = std or [0.229, 0.224, 0.225]
        self.fg_threshold = fg_threshold
        self.bg_threshold = bg_threshold
        self.noise_type = noise_type
        self.noise_std = noise_std
        
 
        self.base_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),  # Reduced
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=5)], p=0.3),  # Reduced
            transforms.RandomGrayscale(p=0.1),  # Reduced
        ])
        
        self.normalize = transforms.Normalize(mean=self.mean, std=self.std)
    
    def __call__(self, image, saliency_map, view_type='foreground'):
        
 
        image = self.base_transform(image)
    

        saliency_transform = transforms.Compose([
        transforms.Resize((self.image_size, self.image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        ])
        saliency_map = saliency_transform(saliency_map)
    
  
        image_tensor = transforms.ToTensor()(image)
        saliency_tensor = transforms.ToTensor()(saliency_map)
    
        if view_type == 'foreground':
            image_tensor = apply_saliency_masking(
            image_tensor, saliency_tensor, 'foreground', 
            self.noise_type, self.noise_std
            )
        elif view_type == 'background':
            image_tensor = apply_saliency_masking(
                image_tensor, saliency_tensor, 'background', 
                self.noise_type, self.noise_std
            )
            

        image_tensor = self.normalize(image_tensor)
    
        return image_tensor, saliency_tensor


# In[ ]:


def saliency_guided_collate_fn(batch):

    images, saliency_maps = zip(*batch)
    
   
    saliency_aug = SaliencyGuidedAugmentation(
        image_size=256,
        fg_threshold=0.6,    
        bg_threshold=0.3,    
        noise_type='mean',   
        noise_std=0.02
    )
    
    fg_views = []      
    bg_views = []      #
    fg_saliency = []
    bg_saliency = []
    
    for img, sal in zip(images, saliency_maps):

        fg_img, fg_sal = saliency_aug(img, sal, 'foreground')
        

        bg_img, bg_sal = saliency_aug(img, sal, 'background')
        
        fg_views.append(fg_img)
        bg_views.append(bg_img)
        fg_saliency.append(fg_sal)
        bg_saliency.append(bg_sal)
    
    return torch.stack(fg_views), torch.stack(bg_views), torch.stack(fg_saliency), torch.stack(bg_saliency)


# In[ ]:




