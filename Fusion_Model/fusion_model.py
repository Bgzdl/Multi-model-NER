from Data_Processing.ReadData import CustomDataset
from Image_Encoder.build_image_encoder import build_image_model
from Text_Encoder.build_text_encoder import build_text_encoder
import torch.nn as nn
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
from torch.cuda.amp import autocast, GradScaler



class FusionModel(nn.Module):
    def __init__(self, image_model_name: str = 'ViT-B/16', text_model_name: str = 'bert-base-cased', proj_dim=512):
        super().__init__()
        self.visual, self.visual_preprocess = build_image_model(image_model_name, proj_dim=proj_dim)
        self.tokenizer, self.text = build_text_encoder(text_model_name, proj_dim=proj_dim)

    def forward(self, image: torch.tensor, text: torch.tensor):
        I = self.visual(image)
        T = self.text(text)
        print(I.shape, T.shape)
        return I


if __name__ == '__main__':
    model = FusionModel()
    device = 'cuda'
    model.to(device)
    # print(model.visual_preprocess)
    dataset = CustomDataset('../../IJCAI2019_data', image_preprocess=model.visual_preprocess, text_preprocess=model.tokenizer, max_len=32)
    print(len(dataset))
    max_len = 0
    dataloader = DataLoader(dataset, batch_size=16)
    scaler = GradScaler()
    with autocast():
        for batch in dataloader:
            image, text, label = batch
            image = image.to(device)
            for key, value in text.items():
                text[key] = value.to(device)
            output = model(image, text)
            break

        # scaler.scale(loss).backward()
        # scaler.step(optimizer)
        # scaler.update()
