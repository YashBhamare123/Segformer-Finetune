from torch.utils.data import Dataset, DataLoader
import datasets
from transformers import SegformerImageProcessor
import torchvision.transforms as T
import torch

class SegmentationDataset(Dataset):
    def __init__(self, hf_ds : datasets.Dataset, processor : SegformerImageProcessor):
        super().__init__()
        self.ds = hf_ds
        self.processor = processor
        self.transform_img = T.Compose([
            T.RandomApply([T.RandomResizedCrop((600, 400), scale=(0.5, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=2),
                        T.RandomRotation(15.)
                        ], p = 0.6)
        ])
        self.transform_post_img = T.Compose([
                        T.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.2)])
        

    def sync_transforms(self, image, mask):
        rng_state = torch.get_rng_state()
        image = self.transform_img(image)
        torch.set_rng_state(rng_state)
        mask = self.transform_img(mask)
        image = self.transform_post_img(image)
        out = {'images' : image, 'segmentation_maps' : mask}
        return out

    
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, idx):
        processed_input = self.processor(**self.sync_transforms(self.ds[idx]['image'], self.ds[idx]['mask']), return_tensors = 'pt')
        processed_input['pixel_values'] = processed_input['pixel_values'].squeeze(0)
        processed_input['labels'] = processed_input['labels'].squeeze(0)
        return processed_input

