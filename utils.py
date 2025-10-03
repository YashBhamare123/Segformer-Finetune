from torch.utils.data import Dataset, DataLoader
import datasets
from transformers import SegformerImageProcessor

class SegmentationDataset(Dataset):
    def __init__(self, hf_ds : datasets.Dataset, processor : SegformerImageProcessor):
        super().__init__()
        self.ds = hf_ds
        self.processor = processor
    
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, idx):
        processed_input = self.processor(images = self.ds[idx]['image'], segmentation_maps = self.ds[idx]['mask'], return_tensors = 'pt')
        processed_input['pixel_values'] = processed_input['pixel_values'].squeeze(0)
        processed_input['labels'] = processed_input['labels'].squeeze(0)
        return processed_input

