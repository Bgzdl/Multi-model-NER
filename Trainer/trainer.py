import torch
import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
from torch.cuda.amp import autocast


class Trainer(object):
    def __init__(self, model: nn.Module, device: str, epochs: int, train_dataloader: DataLoader,
                 valid_dataloader: DataLoader, test_dataloader: DataLoader, optimizer: optim.Optimizer,
                 scheduler, scaler, loss_fun, save_path):
        self.model = model
        self.device = device
        self.epochs = epochs
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.test_dataloader = test_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler
        self.loss_fun = loss_fun
        self.save_path = save_path

    @staticmethod
    def calculate_loss(outputs, labels, sentence_len, loss_fun):
        """
        :param outputs: [batch size, max_seq_length, categories]
        :param labels: [batch size, max_seq_length, categories]
        :param sentence_len: [batch size]
        :param loss_fun: nn.CrossEntropyLoss
        :return: loss
        """
        loss = 0.0
        batch_size = outputs.shape[0]
        for i in range(outputs.shape[0]):
            output = outputs[i][:sentence_len[i]]
            label = labels[i][:sentence_len[i]].float()
            # print(output.shape, label.shape)
            current_loss = loss_fun(output, label)
            # print(i, current_loss.item(), sentence_len[i])
            loss += current_loss / sentence_len[i]
        return loss / batch_size

    @staticmethod
    def predict(output):
        """
        :param output: [batch size, max_seq_length, categories]
        :return: [batch size, max_seq_length]
        """
        indices = torch.argmax(output, dim=-1, keepdim=True)
        predict = torch.zeros_like(output)
        predict.scatter_(-1, indices, 1)
        return predict

    @staticmethod
    def evaluate(predict, label, sentence_len):
        """
        :param predict: [batch size, max_seq_len, categories]
        :param label: [batch size, max_seq_len, categories]
        :param sentence_len: [batch size, 1], type is list
        :return:
        """
        total_num = sum(sentence_len)
        batch_size = predict.shape[0]
        correct = 0
        for i in range(batch_size):
            sample_predict = predict[i][:sentence_len[i]]
            sample_label = label[i][:sentence_len[i]]
            equal_vectors = torch.all(sample_predict == sample_label, dim=-1)
            # 计算相等的 one-hot 编码向量的数量
            correct += torch.sum(equal_vectors).item()
        return correct / total_num, correct, total_num

    def train(self):
        loss_epoch = []
        acc_epoch = []
        best_epoch = 0
        for epoch in range(self.epochs):
            # 训练
            self.model.train()
            train_bar = tqdm(self.train_dataloader, desc=f"Train epoch {epoch + 1}/{self.epochs}")
            running_loss = 0.0
            for batch in train_bar:
                with autocast():
                    image, text, label, sentence_len = batch
                    # label: max_sentence_len * batch size
                    image = image.to(self.device)
                    for key, value in text.items():
                        text[key] = value.to(self.device)
                    label = label.to(self.device)
                    output = self.model(image, text)
                    # 测试验证代码
                    # 测试结束
                    loss = self.calculate_loss(output, label, sentence_len, self.loss_fun)
                    running_loss += loss.item()
                    train_bar.set_postfix(loss=loss.item())
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
            loss_epoch.append(running_loss/len(train_bar))
            # 验证
            self.model.eval()
            val_bar = tqdm(self.valid_dataloader, desc=f"Valid epoch {epoch + 1}/{self.epochs}")
            correct, total = 0, 0
            with torch.no_grad():
                for batch in val_bar:
                    image, text, label, sentence_len = batch
                    # label: max_sentence_len * batch size
                    image = image.to(self.device)
                    for key, value in text.items():
                        text[key] = value.to(self.device)
                    label = label.to(self.device)
                    with autocast():
                        output = self.model(image, text)
                        predict = self.predict(output)
                        print(predict.shape)
                        print(predict[0][0])
                        acc, correct_num, total_num = self.evaluate(predict, label, sentence_len)
                        correct += correct_num
                        total += total_num
                        val_bar.set_postfix(acc=acc)
                total_acc = correct / total
                acc_epoch.append(total_acc)
                if total_acc > max(acc_epoch) and not epoch == 0:
                    self.save_model(f'Best_model_from_{epoch + 1}')
                    best_epoch = epoch + 1
                    print(f"Best model saved ,epoch is {epoch + 1}")
        # 日志保存
        self.save_log(loss_epoch, acc_epoch, best_epoch)

    def test(self):
        self.model.eval()
        with torch.no_grad():
            test_bar = tqdm(self.test_dataloader, desc=f"Testing")
            correct, total = 0, 0
            for batch in test_bar:
                image, text, label, sentence_len = batch
                # label: max_sentence_len * batch size
                image = image.to(self.device)
                for key, value in text.items():
                    text[key] = value.to(self.device)
                label = label.to(self.device)
                with autocast():
                    output = self.model(image, text)
                    predict = self.predict(output)
                    acc, correct_num, total_num = self.evaluate(predict, label, sentence_len)
                    correct += correct_num
                    total += total_num
                    test_bar.set_postfix(acc=acc)
            total_acc = correct / total
            file_path = self.save_path + 'Test_result.txt'
            with open(file_path, 'w') as file:
                file.write(f'Final accuracy is {total_acc}')
            file.close()
            print(f"Test is over, final accuracy is {total_acc}")

    def save_model(self, model_name):
        torch.save(self.model.state_dict(), self.save_path + model_name)
        print("Model save successfully")

    def save_log(self, loss: list, acc: list, best_epoch: int):
        data = {"loss": loss, 'acc': acc, 'best_epoch': [best_epoch] + [None] * (len(loss) - 1)}
        df = pd.DataFrame(data)
        df.to_csv(self.save_path + 'Train_log.csv', index=False)
        print('Log save successfully')
