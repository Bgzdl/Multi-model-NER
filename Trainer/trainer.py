import torch
import os
import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
from torch.cuda.amp import autocast
from utils import *


class Trainer(object):
    def __init__(
            self,
            model: nn.Module,
            device: str,
            epochs: int,
            train_dataloader: DataLoader,
            valid_dataloader: DataLoader,
            test_dataloader: DataLoader,
            optimizer: optim.Optimizer,
            scheduler,
            scaler,
            loss_fun,
            save_path
    ):
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
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    @staticmethod
    def evaluate(text, tokenizer, predict, label, sentence_len):
        """
        :param predict: list([batch size, seq_len])
        :param label: tensor([batch size, max_seq_len])
        :param sentence_len: list([batch size, 1])
        :return:
        """
        total_num = sum(sentence_len)
        batch_size = len(predict)
        correct_per_position = 0

        correct_ne = 0
        total_pred_entities = 0
        total_gold_entities = 0

        token_idxs = text['input_ids']
        tokenized_texts = extract_token(token_idxs, tokenizer)

        for i in range(batch_size):
            tokenized_text = tokenized_texts[i]
            sample_predict = torch.tensor(predict[i], device=label.device)[1: -1]
            sample_label = label[i][1: sentence_len[i] - 1]
            if sample_predict.shape != sample_label.shape:
                sample_label = sample_label[:len(sample_predict)]
            assert sample_predict.shape == sample_label.shape
            equal_vectors = torch.eq(sample_predict, sample_label)
            correct_per_position += torch.sum(equal_vectors).item()

            pred_named_entity = extract_named_entity(sample_predict, tokenized_text)
            gold_named_entity = extract_named_entity(sample_label, tokenized_text)

            correct_predictions, num_pred_entities, num_gold_entities = calculate_ner_accuracy(gold_named_entity,
                                                                                               pred_named_entity)
            correct_ne += correct_predictions
            total_pred_entities += num_pred_entities
            total_gold_entities += num_gold_entities

        acc = correct_per_position / total_num if total_num != 0 else 0.0
        precision = correct_ne / total_pred_entities if total_pred_entities != 0 else 0.0
        recall = correct_ne / total_gold_entities if total_gold_entities != 0 else 0.0

        return {
            'macro_metric': [acc, precision, recall],
            'micro_metric': [
                correct_per_position, total_num,  # acc metric
                correct_ne, total_pred_entities, total_gold_entities  # f1 metric
            ]
        }

    def train(self):
        loss_epoch = []
        acc_epoch = []
        precision_epoch = []
        recall_epoch = []
        f1_epoch = []
        best_epoch = 0
        for epoch in range(self.epochs):
            # 训练
            self.model.train()
            train_bar = tqdm(self.train_dataloader, desc=f"Train epoch {epoch + 1}/{self.epochs}")
            total_loss = 0.0
            for batch in train_bar:
                with autocast():
                    image, text, label, _ = batch

                    image = image.to(self.device)
                    for key, value in text.items():
                        text[key] = value.to(self.device)
                    label = label.to(self.device)

                    loss = self.model(image, text, label)
                    total_loss += loss.item()

                    train_bar.set_postfix(loss=loss.item())

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            total_loss = total_loss / len(self.train_dataloader)
            loss_epoch.append(total_loss)

            # 验证
            self.model.eval()
            val_bar = tqdm(self.valid_dataloader, desc=f"Valid epoch {epoch + 1}/{self.epochs}")
            correct_pre_position, total_token = 0, 0
            correct_ne, total_pred_entities, total_gold_entities = 0, 0, 0
            macro_acc, macro_precision, macro_recall = 0, 0, 0
            with torch.no_grad():
                for batch in val_bar:
                    image, text, label, sentence_len = batch
                    # transfer to device
                    image = image.to(self.device)
                    for key, value in text.items():
                        text[key] = value.to(self.device)
                    label = label.to(self.device)

                    with autocast():
                        output = self.model(image, text, None)
                        metric = self.evaluate(text, self.model.tokenizer, output, label, sentence_len)
                        correct_pre_position += metric['micro_metric'][0]
                        total_token += metric['micro_metric'][1]
                        correct_ne += metric['micro_metric'][2]
                        total_pred_entities += metric['micro_metric'][3]
                        total_gold_entities += metric['micro_metric'][4]

                    macro_acc += metric['macro_metric'][0]
                    macro_precision += metric['macro_metric'][1]
                    macro_recall += metric['macro_metric'][2]

                    val_bar.set_postfix(
                        acc=metric['macro_metric'][0],
                        precision=metric['macro_metric'][1],
                        recall=metric['macro_metric'][2]
                    )

            micro_acc = correct_pre_position / total_token
            micro_precision = correct_ne / total_pred_entities
            micro_recall = correct_ne / total_gold_entities
            micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall)

            macro_acc /= len(self.valid_dataloader)
            macro_precision /= len(self.valid_dataloader)
            macro_recall /= len(self.valid_dataloader)
            macro_f1 = 2 * (macro_precision * macro_recall) / (macro_precision + macro_recall)

            print(
                f"Epoch {epoch + 1}/{self.epochs}, "
                f"micro_acc: {micro_acc}, micro_precision: {micro_precision}, micro_recall: {micro_recall}, micro_f1: {micro_f1}"
                f"macro_acc: {macro_acc}, macro_precision: {macro_precision}, macro_recall: {macro_recall}, macro_f1: {macro_f1}"
            )

            acc_epoch.append(micro_acc)
            precision_epoch.append(micro_precision)
            recall_epoch.append(micro_recall)
            f1_epoch.append(micro_f1)

            if micro_f1 >= max(f1_epoch):
                self.save_model(f'Best_model_from_{epoch + 1}')
                best_epoch = epoch + 1
                print(f"Best model saved ,epoch is {epoch + 1}")
        # 日志保存
        self.save_log(loss_epoch, acc_epoch, precision_epoch, recall_epoch, f1_epoch, best_epoch)

    def test(self):
        self.model.eval()
        with torch.no_grad():
            test_bar = tqdm(self.test_dataloader, desc=f"Testing")
            correct_pre_position, total_token = 0, 0
            correct_ne, total_pred_entities, total_gold_entities = 0, 0, 0
            macro_acc, macro_precision, macro_recall = 0, 0, 0
            for batch in test_bar:
                image, text, label, sentence_len = batch

                image = image.to(self.device)
                for key, value in text.items():
                    text[key] = value.to(self.device)
                label = label.to(self.device)

                with autocast():
                    output = self.model(image, text, None)
                    metric = self.evaluate(output, label, sentence_len)
                    correct_pre_position += metric['micro_metric'][0]
                    total_token += metric['micro_metric'][1]
                    correct_ne += metric['micro_metric'][2]
                    total_pred_entities += metric['micro_metric'][3]
                    total_gold_entities += metric['micro_metric'][4]

                macro_acc += metric['macro_metric'][0]
                macro_precision += metric['macro_metric'][1]
                macro_recall += metric['macro_metric'][2]

            micro_acc = correct_pre_position / total_token
            micro_precision = correct_ne / total_pred_entities
            micro_recall = correct_ne / total_gold_entities
            micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall)

            macro_acc /= len(self.valid_dataloader)
            macro_precision /= len(self.valid_dataloader)
            macro_recall /= len(self.valid_dataloader)
            macro_f1 = 2 * (macro_precision * macro_recall) / (macro_precision + macro_recall)

            file_path = self.save_path + 'Test_result.txt'
            with open(file_path, 'w') as file:
                file.write(
                    f"micro_acc: {micro_acc}, micro_precision: {micro_precision}, micro_recall: {micro_recall}, micro_f1: {micro_f1}. "
                    f"macro_acc: {macro_acc}, macro_precision: {macro_precision}, macro_recall: {macro_recall}, macro_f1: {macro_f1}"
                )
            file.close()
            print("write to Test_result.txt successfully")

    def save_model(self, model_name):
        torch.save(self.model.state_dict(), self.save_path + model_name)
        print("Model save successfully")

    def save_log(self, loss_epoch, acc_epoch, precision_epoch, recall_epoch, f1_epoch, best_epoch):
        data = {
            'loss': loss_epoch,
            'acc': acc_epoch,
            'precision': precision_epoch,
            'recall': recall_epoch,
            'f1': f1_epoch,
            'best_epoch': [best_epoch] + [None] * (len(loss_epoch) - 1)
        }
        df = pd.DataFrame(data)
        df.to_csv(self.save_path + 'Train_log.csv', index=False)
        print('Log save successfully')
