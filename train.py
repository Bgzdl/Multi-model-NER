import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from Fusion_Model.fusion_model import FusionModel
from Data_Processing.ReadData import CustomDataset
from torch.cuda.amp import GradScaler
from Trainer.trainer import Trainer


def calculate_loss(outputs, labels, sentence_len, loss_fun):
    loss = 0.0
    batch_size = outputs.shape[0]
    for i in range(outputs.shape[0]):
        output = outputs[i][:sentence_len[i]]
        label = labels[i][:sentence_len[i]].float()
        current_loss = loss_fun(output, label)
        loss += current_loss / sentence_len[i]
    return loss / batch_size


if __name__ == "__main__":
    # 参数设定
    max_seq_len = 64
    device = 'cuda'
    epochs = 30
    # 模型准备
    model = FusionModel(max_seq_len=max_seq_len, device=device, text_model_name='distilbert-NER')
    model.to(device)
    # 数据准备
    print("Preparing Data ...")
    train_dataset = CustomDataset('../IJCAI2019_data', mode='train', image_preprocess=model.visual_preprocess, text_preprocess=model.tokenizer, max_len=max_seq_len)
    train_dataloader = DataLoader(train_dataset, batch_size=16)
    valid_dataset = CustomDataset('../IJCAI2019_data', mode='valid', image_preprocess=model.visual_preprocess, text_preprocess=model.tokenizer, max_len=max_seq_len)
    valid_dataloader = DataLoader(valid_dataset, batch_size=16)
    test_dataset = CustomDataset('../IJCAI2019_data', mode='test', image_preprocess=model.visual_preprocess, text_preprocess=model.tokenizer, max_len=max_seq_len)
    test_dataloader = DataLoader(test_dataset, batch_size=16)
    print()
    # 损失函数，优化器，混合进度scaler，学习率调度器
    scaler = GradScaler()
    loss_fun = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    trainer = Trainer(model, device, epochs, train_dataloader, valid_dataloader, test_dataloader, optimizer, scheduler, scaler, loss_fun, save_path='./result/')
    trainer.train()
    trainer.test()
    # # 此处一定要使用混合精度进行训练
    # for epoch in range(epochs):
    #     running_loss = 0.0
    #     model.train()
    #     with autocast():
    #         progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")
    #         for batch in progress_bar:
    #             image, text, label, sentence_len = batch
    #             # label: max_sentence_len * batch size
    #             image = image.to(device)
    #             for key, value in text.items():
    #                 text[key] = value.to(device)
    #             label = label.to(device)
    #             output = model(image, text)
    #             loss = calculate_loss(output, label, sentence_len, loss_fun)
    #             running_loss += loss.item()
    #             progress_bar.set_postfix(loss=loss.item())
    #             # print(loss.item())
    #             scaler.scale(loss).backward()
    #             scaler.step(optimizer)
    #             scaler.update()
    #             # 更新学习率
    #             scheduler.step()
    #         print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {running_loss / len(dataloader)}")
