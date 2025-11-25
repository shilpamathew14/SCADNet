#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from torch.utils.data import Dataset


class IDRIDDirectPatchDataset(Dataset):
    def __init__(self, image_patches_dir, mask_patches_dir, transform=None):
        
        self.image_patches_dir = image_patches_dir
        self.mask_patches_dir = mask_patches_dir
        self.transform = transform

        self.image_filenames = sorted(os.listdir(image_patches_dir))
        self.mask_filenames = sorted(os.listdir(mask_patches_dir))  # Assumes same order

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_patches_dir, self.image_filenames[idx])
        mask_path = os.path.join(self.mask_patches_dir, self.mask_filenames[idx])

        # Load image and mask
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')  # Grayscale mask

        image = np.array(image)
        mask = np.array(mask).astype(np.uint8)

        if self.transform:
            if hasattr(self.transform, 'replay_mode'):
                transformed = self.transform(image=image, mask=mask)
                image = transformed['image']
                mask = transformed['mask']
            else:
                image = self.transform(Image.fromarray(image))
                mask = torch.from_numpy(mask).long()
        else:
            image = transforms.ToTensor()(image)
            mask = torch.from_numpy(mask).long()

        return image, mask

