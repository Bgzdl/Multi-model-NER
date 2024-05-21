from Data_Processing.ReadData import CustomDataset
from Image_Encoder.build_image_encoder import build_image_model, build_Lora_image_model
from Text_Encoder.build_text_encoder import build_text_encoder
import torch.nn as nn
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
from torch.cuda.amp import autocast, GradScaler
from Pipline.pipline import Pipline
from Fusion_Model.fusion_layer import FusionBlock


class FusionModel(nn.Module):
    def __init__(self, image_model_name: str = 'ViT-B/16', text_model_name: str = 'bert-base-cased', proj_dim=512, num_of_heads=8, max_seq_len=64, device='cpu'):
        super().__init__()
        self.visual, self.visual_preprocess = build_Lora_image_model(image_model_name, proj_dim=proj_dim)
        self.tokenizer, self.text = build_text_encoder(text_model_name, proj_dim=proj_dim, frozen=True)
        self.fusion_block = FusionBlock(3, proj_dim, num_of_heads, 0.5)
        self.fusion_block.to(device)
        self.pipeline = Pipline(proj_dim, max_seq_len, 9)

    def forward(self, image: torch.tensor, text: torch.tensor):
        I = self.visual(image)
        T = self.text(text)
        I = I.unsqueeze(1)
        I = I.expand([-1, T.shape[1], -1])
        # I:[batch size, max_seq_length, dimension], T:[batch size, max_seq_length, dimension]
        fusion_embedding = self.fusion_block(I, T)
        output = self.pipeline(fusion_embedding)
        return output


def calculate_loss(outputs, labels, sentence_len, loss_fun):
    loss = 0
    for i in range(outputs.shape[0]):
        output = outputs[i][:sentence_len[i]]
        label = labels[i][:sentence_len[i]].float()
        print(output.shape, label.shape)
        loss += loss_fun(output, label)
        break
    return loss


if __name__ == '__main__':
    max_seq_len = 64
    device = 'cuda'
    model = FusionModel(max_seq_len=max_seq_len, device=device)
    model.to(device)
    dataset = CustomDataset('../../IJCAI2019_data', image_preprocess=model.visual_preprocess, text_preprocess=model.tokenizer, max_len=max_seq_len)
    max_len = 0
    dataloader = DataLoader(dataset, batch_size=16)
    scaler = GradScaler()
    loss = nn.CrossEntropyLoss()
    # 此处一定要使用混合精度进行训练
    with autocast():
        for batch in dataloader:
            image, text, label, sentence_len = batch
            # label: max_sentence_len * batch size
            image = image.to(device)
            for key, value in text.items():
                text[key] = value.to(device)
            label = label.to(device)
            output = model(image, text)
            loss = calculate_loss(output, label, sentence_len, loss)
            break

        # scaler.scale(loss).backward()
        # scaler.step(optimizer)
        # scaler.update()
