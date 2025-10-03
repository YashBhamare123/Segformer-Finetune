from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
from datasets import load_dataset
from PIL import Image
import requests
import matplotlib.pyplot as plt
import torch.nn as nn
from torchvision.transforms import ToTensor, ToPILImage
import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from tqdm import tqdm

from utils import SegmentationDataset


def train(epochs : int,
           model,
           train_dl : DataLoader,
           val_dl : DataLoader,
           optim : torch.optim.AdamW,
           writer : SummaryWriter,
           val_size : int,
           train_size : int,
           device : str
        ):
    
    for epoch in tqdm(range(epochs)):

        mean_train_loss = 0
        model.train()
        for inps in train_dl:
            inps['pixel_values'] = inps['pixel_values'].to(device = device)
            inps['labels'] = inps['labels'].to(device = device)

            out = model(**inps)
            loss = out.loss
            optim.zero_grad()
            loss.backward()
            optim.step()

            mean_train_loss += loss.item()

        mean_train_loss = mean_train_loss / train_size
        writer.add_scalar('Training Loss', mean_train_loss, global_step = epoch)

        mean_val_loss = 0    
        model.eval()
        for idx, inps in enumerate(val_dl):
            inps['pixel_values'] = inps['pixel_values'].to(device = device)
            inps['labels'] = inps['labels'].to(device = device)

            out = model(**inps)
            val_loss = out.loss
            logits = out.logits.detach().cpu()
            mean_val_loss += val_loss.item()

            if (4 * idx % val_size == 0):
                upsampled_logits = F.interpolate(
                    input = logits,
                    size = (600, 400),
                    mode = 'bilinear',
                    align_corners= False
                )
                pred_seg = torch.argmax(upsampled_logits, dim = 1)
                pred_seg = pred_seg.to(dtype = torch.float32) / 255.
                pred_seg = pred_seg.unsqueeze(1)
                mask_grid = make_grid(pred_seg)
                writer.add_image('Predicted Masks', mask_grid, global_step= 4 * epoch + 4 * idx // val_size)
            
        mean_val_loss = mean_val_loss / val_size
        writer.add_scalar('Validation Loss', mean_val_loss, global_step = epoch)


def main():
    # dataset and preprocessor load
    ds = load_dataset('YashBhamare123/human_parsing_dataset_plus_neck', split = 'train[:1000]')
    processor = SegformerImageProcessor.from_pretrained('mattmdjaga/segformer_b2_clothes')

    dataset = SegmentationDataset(ds, processor)
    train_ds, val_ds = random_split(dataset, [int(0.8 * len(dataset)), int(0.2 * len(dataset))])
    train_dl = DataLoader(train_ds, batch_size = 16, shuffle = True, pin_memory= True)
    val_dl = DataLoader(val_ds, batch_size = 16, shuffle = True, pin_memory = True)

    # model prep
    model = AutoModelForSemanticSegmentation.from_pretrained('mattmdjaga/segformer_b2_clothes')
    device = 'cuda'
    for name, params in model.named_parameters():
        if 'decode' not in name:
            params.requires_grad = False
    
    model.decode_head.classifier = nn.Conv2d(768, 19, 1, 1, 0)
    model.to(device = device)

    # train prep
    epochs = 100
    lr = 1e-8
    optim = torch.optim.AdamW(model.parameters(), lr = lr)
    train_size = len(train_ds)
    val_size = len(val_ds)
    writer = SummaryWriter(log_dir = './runs')

    train(epochs, model, train_dl, val_dl, optim, writer, val_size, train_size, device)


if __name__ == '__main__':
    main()
    



    


















