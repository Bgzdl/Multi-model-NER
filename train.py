import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from Fusion_Model.fusion_model import FusionModel
from Data_Processing.ReadData import CustomDataset
from torch.cuda.amp import GradScaler
from Trainer.trainer import Trainer


def get_weight(dataset):
    num_of_label = torch.zeros_like(dataset[0][2][0], dtype=torch.float32)
    for i, data in enumerate(dataset):
        _, _, label, _ = data
        current_label = torch.sum(label, dim=0)
        num_of_label += current_label
    # 取反比
    weight = 1 / num_of_label
    # 归一化
    weight = weight / weight.sum()
    return weight


if __name__ == "__main__":
    # 参数设定
    max_seq_len = 128
    device = 'cuda'
    epochs = 30
    # 模型准备
    model = FusionModel(max_seq_len=max_seq_len, device=device, text_model_name='bert-base-cased')
    model.to(device)
    # for name, param in model.named_parameters():
    #     print(f"Layer: {name} | Frozen: {not param.requires_grad}")
    # # 数据准备
    print("Preparing Data ...")
    train_dataset = CustomDataset('../IJCAI2019_data', mode='train', image_preprocess=model.visual_preprocess, text_preprocess=model.tokenizer, max_len=max_seq_len)
    train_dataloader = DataLoader(train_dataset, batch_size=16)
    valid_dataset = CustomDataset('../IJCAI2019_data', mode='valid', image_preprocess=model.visual_preprocess, text_preprocess=model.tokenizer, max_len=max_seq_len)
    valid_dataloader = DataLoader(valid_dataset, batch_size=16)
    test_dataset = CustomDataset('../IJCAI2019_data', mode='test', image_preprocess=model.visual_preprocess, text_preprocess=model.tokenizer, max_len=max_seq_len)
    test_dataloader = DataLoader(test_dataset, batch_size=16)
    # 统计训练集中各个样本的类别数量，以数量为反比作为权重
    weight = get_weight(train_dataset)
    # 损失函数，优化器，混合进度scaler，学习率调度器
    scaler = GradScaler()
    loss_fun = nn.CrossEntropyLoss(weight=weight)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    trainer = Trainer(model, device, epochs, train_dataloader, valid_dataloader, test_dataloader, optimizer, scheduler, scaler, loss_fun, save_path='./result/')
    trainer.train()
    trainer.test()
