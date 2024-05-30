import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from Fusion_Model.fusion_model import FusionModel
from Data_Processing.ReadData import CustomDataset
from torch.cuda.amp import GradScaler
from Trainer.trainer import Trainer


if __name__ == "__main__":
    # 参数设定
    max_seq_len = 64
    device = 'cuda'
    epochs = 5
    # 模型准备
    model = FusionModel(num_classes=9).to(device)
    # 数据准备
    print("Preparing Data ...")
    train_dataset = CustomDataset('./datas/IJCAI2019_data', mode='train', image_preprocess=model.visual_preprocess, text_preprocess=model.tokenizer, max_len=max_seq_len)
    train_dataloader = DataLoader(train_dataset, batch_size=16)
    valid_dataset = CustomDataset('./datas/IJCAI2019_data', mode='valid', image_preprocess=model.visual_preprocess, text_preprocess=model.tokenizer, max_len=max_seq_len)
    valid_dataloader = DataLoader(valid_dataset, batch_size=16)
    test_dataset = CustomDataset('./datas/IJCAI2019_data', mode='test', image_preprocess=model.visual_preprocess, text_preprocess=model.tokenizer, max_len=max_seq_len)
    test_dataloader = DataLoader(test_dataset, batch_size=16)
    # 损失函数，优化器，混合进度scaler，学习率调度器
    scaler = GradScaler()
    loss_fun = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    trainer = Trainer(
        model,
        device,
        epochs,
        train_dataloader,
        valid_dataloader,
        test_dataloader,
        optimizer,
        scheduler,
        scaler,
        loss_fun,
        save_path='./result/'
    )

    trainer.train()

    # trainer.test()

