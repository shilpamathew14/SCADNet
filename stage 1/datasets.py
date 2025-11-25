#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
from torch.utils.data import Dataset

class FundusDataset(Dataset):
    def __init__(self, root_dir, sal_dir):
        self.root_dir = root_dir
        self.sal_dir = sal_dir
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        img_name = os.path.splitext(self.image_files[idx])[0]
        
        from PIL import Image
        from torchvision import transforms
        
        image = Image.open(img_path).convert("RGB")
        image = transforms.Resize((256, 256))(image)

        saliency_image = img_name + "_saliency.jpg"
        saliency_path = os.path.join(self.sal_dir, saliency_image)

        if not os.path.exists(saliency_path):
            raise FileNotFoundError(f"Saliency map not found: {saliency_path}")

        saliency_map = Image.open(saliency_path).convert('L')
        saliency_map = transforms.Resize((256, 256))(saliency_map)
        
        return image, saliency_map

