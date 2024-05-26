import torch
import torch.nn as nn
from Image_Encoder.build_image_encoder import build_image_model
from Text_Encoder.build_text_encoder import build_text_encoder
from Fusion_Model.fusion_layer import FusionBlock
from Fusion_Model.pipeline import Pipeline
from torchcrf import CRF


class FusionModel(nn.Module):
    def __init__(
            self,
            image_model_name: str = 'ViT-B/16',
            text_model_name: str = 'bert-base-cased',
            proj_dim=512,
            num_of_heads=8,
            num_classes=9
    ):
        super().__init__()

        self.visual, self.visual_preprocess = build_image_model(image_model_name, proj_dim=proj_dim)
        self.tokenizer, self.text = build_text_encoder(text_model_name, proj_dim=proj_dim)
        self.fusion_block = FusionBlock(1, proj_dim, num_of_heads, 0.5)
        self.pipeline = Pipeline(proj_dim, num_classes)
        self.crf = CRF(num_classes, batch_first=True)

    def forward(self, image, text, label):
        I = self.visual(image)
        T = self.text(text)
        I = I.unsqueeze(1)
        I = I.expand([-1, T.shape[1], -1])

        # I:[batch size, max_seq_length, dimension], T:[batch size, max_seq_length, dimension]
        fusion_embedding = self.fusion_block(I, T)

        output = self.pipeline(fusion_embedding)

        if self.training:
            loss = -self.crf(output, label, mask=label.gt(-1))
            return loss
        else:
            seq = self.crf.decode(output, mask=text['attention_mask'].to(torch.bool))
            return seq


if __name__ == '__main__':
    from Data_Processing.ReadData import CustomDataset
    from torch.utils.data import DataLoader
    from torch.cuda.amp import autocast, GradScaler

    max_seq_len = 64
    device = 'cuda'
    model = FusionModel().to(device)
    dataset = CustomDataset(
        '../datas/IJCAI2019_data',
        mode='train',
        image_preprocess=model.visual_preprocess,
        text_preprocess=model.tokenizer,
        max_len=max_seq_len
    )
    dataloader = DataLoader(dataset, batch_size=2)
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
            loss = model(image, text, label)
            print(loss)
            break
