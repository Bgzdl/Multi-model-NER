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

        per_correct_ne = 0
        per_total_pred_entities = 0
        per_total_gold_entities = 0

        loc_correct_ne = 0
        loc_total_pred_entities = 0
        loc_total_gold_entities = 0

        org_correct_ne = 0
        org_total_pred_entities = 0
        org_total_gold_entities = 0
        
        misc_correct_ne = 0
        misc_total_pred_entities = 0
        misc_total_gold_entities = 0
        
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
            per_correct_predictions, per_num_pred_entities, per_num_gold_entities = calculate_per_accuracy(gold_named_entity,
                                                                                               pred_named_entity)
            loc_correct_predictions, loc_num_pred_entities, loc_num_gold_entities = calculate_loc_accuracy(gold_named_entity,
                                                                                               pred_named_entity)            
            org_correct_predictions, org_num_pred_entities, org_num_gold_entities = calculate_org_accuracy(gold_named_entity,
                                                                                               pred_named_entity)            
            misc_correct_predictions, misc_num_pred_entities, misc_num_gold_entities = calculate_misc_accuracy(gold_named_entity,
                                                                                               pred_named_entity)            
            
            correct_ne += correct_predictions
            total_pred_entities += num_pred_entities
            total_gold_entities += num_gold_entities
            
            per_correct_ne += per_correct_predictions
            per_total_pred_entities += per_num_pred_entities
            per_total_gold_entities += per_num_gold_entities
            
            loc_correct_ne += loc_correct_predictions
            loc_total_pred_entities += loc_num_pred_entities
            loc_total_gold_entities += loc_num_gold_entities
            
            org_correct_ne += org_correct_predictions
            org_total_pred_entities += org_num_pred_entities
            org_total_gold_entities += org_num_gold_entities  
            
            misc_correct_ne += misc_correct_predictions
            misc_total_pred_entities += misc_num_pred_entities
            misc_total_gold_entities += misc_num_gold_entities
            
        acc = correct_per_position / total_num if total_num != 0 else 0.0
        precision = correct_ne / total_pred_entities if total_pred_entities != 0 else 0.0
        recall = correct_ne / total_gold_entities if total_gold_entities != 0 else 0.0
        
        per_precision = per_correct_ne / per_total_pred_entities if per_total_pred_entities != 0 else 0.0
        per_recall = per_correct_ne / per_total_gold_entities if per_total_gold_entities != 0 else 0.0    
        
        loc_precision = loc_correct_ne / loc_total_pred_entities if loc_total_pred_entities != 0 else 0.0
        loc_recall = loc_correct_ne / loc_total_gold_entities if loc_total_gold_entities != 0 else 0.0

        org_precision = org_correct_ne / org_total_pred_entities if org_total_pred_entities != 0 else 0.0
        org_recall = org_correct_ne / org_total_gold_entities if org_total_gold_entities != 0 else 0.0
        
        misc_precision = misc_correct_ne / misc_total_pred_entities if misc_total_pred_entities != 0 else 0.0
        misc_recall = misc_correct_ne / misc_total_gold_entities if misc_total_gold_entities != 0 else 0.0        
        
        return {
            'macro_metric': [acc, precision, recall],
            
            'micro_metric': [
                correct_per_position, total_num,  # acc metric
                correct_ne, total_pred_entities, total_gold_entities,  # f1 metric
                per_correct_ne, per_total_pred_entities, per_total_gold_entities,
                loc_correct_ne, loc_total_pred_entities, loc_total_gold_entities,
                org_correct_ne, org_total_pred_entities, org_total_gold_entities,
                misc_correct_ne, misc_total_pred_entities, misc_total_gold_entities,
            ]
                      
        }

    def train(self):
        loss_epoch = []
        acc_epoch = []
        precision_epoch = []
        recall_epoch = []
        f1_epoch = []
        
        per_precision_epoch = []
        per_recall_epoch = []
        per_f1_epoch = []
        
        loc_precision_epoch = []
        loc_recall_epoch = []
        loc_f1_epoch = []
        
        org_precision_epoch = []
        org_recall_epoch = []
        org_f1_epoch = []
        
        misc_precision_epoch = []
        misc_recall_epoch = []
        misc_f1_epoch = []
        
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
            
            per_correct_ne = 0
            per_total_pred_entities = 0
            per_total_gold_entities = 0

            loc_correct_ne = 0
            loc_total_pred_entities = 0
            loc_total_gold_entities = 0

            org_correct_ne = 0
            org_total_pred_entities = 0
            org_total_gold_entities = 0

            misc_correct_ne = 0
            misc_total_pred_entities = 0
            misc_total_gold_entities = 0            
            
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
                        
                        per_correct_ne += metric['micro_metric'][5]
                        per_total_pred_entities += metric['micro_metric'][6]
                        per_total_gold_entities += metric['micro_metric'][7]
                        
                        loc_correct_ne += metric['micro_metric'][8]
                        loc_total_pred_entities += metric['micro_metric'][9]
                        loc_total_gold_entities += metric['micro_metric'][10]
                        
                        org_correct_ne += metric['micro_metric'][11]
                        org_total_pred_entities += metric['micro_metric'][12]
                        org_total_gold_entities += metric['micro_metric'][13]
                        
                        misc_correct_ne += metric['micro_metric'][14]
                        misc_total_pred_entities += metric['micro_metric'][15]
                        misc_total_gold_entities += metric['micro_metric'][16]

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

            per_micro_precision = per_correct_ne / per_total_pred_entities if per_total_pred_entities != 0 else 0.0
            per_micro_recall = per_correct_ne / per_total_gold_entities if per_total_gold_entities != 0 else 0.0 
            per_micro_f1 = 2 * (per_micro_precision * per_micro_recall) / (per_micro_precision + per_micro_recall) if per_micro_precision + per_micro_recall != 0 else 0.0 

            loc_micro_precision = loc_correct_ne / loc_total_pred_entities if loc_total_pred_entities != 0 else 0.0
            loc_micro_recall = loc_correct_ne / loc_total_gold_entities if loc_total_gold_entities != 0 else 0.0
            loc_micro_f1 = 2 * (loc_micro_precision * loc_micro_recall) / (loc_micro_precision + loc_micro_recall) if loc_micro_precision + loc_micro_recall != 0 else 0.0 

            
            org_micro_precision = org_correct_ne / org_total_pred_entities if org_total_pred_entities != 0 else 0.0
            org_micro_recall = org_correct_ne / org_total_gold_entities if org_total_gold_entities != 0 else 0.0
            org_micro_f1 = 2 * (org_micro_precision * org_micro_recall) / (org_micro_precision + org_micro_recall) if org_micro_precision + org_micro_recall != 0 else 0.0 

            
            misc_micro_precision = misc_correct_ne / misc_total_pred_entities if misc_total_pred_entities != 0 else 0.0
            misc_micro_recall = misc_correct_ne / misc_total_gold_entities if misc_total_gold_entities != 0 else 0.0 
            misc_micro_f1 = 2 * (misc_micro_precision * misc_micro_recall) / (misc_micro_precision + misc_micro_recall) if misc_micro_precision + misc_micro_recall != 0 else 0.0 


            macro_acc /= len(self.valid_dataloader)
            macro_precision /= len(self.valid_dataloader)
            macro_recall /= len(self.valid_dataloader)
            macro_f1 = 2 * (macro_precision * macro_recall) / (macro_precision + macro_recall)

            print(
                f"Epoch {epoch + 1}/{self.epochs}, "
                f"micro_acc: {micro_acc}, micro_precision: {micro_precision}, micro_recall: {micro_recall}, micro_f1: {micro_f1}."
                f"macro_acc: {macro_acc}, macro_precision: {macro_precision}, macro_recall: {macro_recall}, macro_f1: {macro_f1}."
                f"PER_micro_precision: {per_micro_precision}, PER_micro_recall: {per_micro_recall}, PER_micro_f1: {per_micro_f1}."
                f"LOC_micro_precision: {loc_micro_precision}, LOC_micro_recall: {loc_micro_recall}, LOC_micro_f1: {loc_micro_f1}."
                f"ORG_micro_precision: {org_micro_precision}, ORG_micro_recall: {org_micro_recall}, ORG_micro_f1: {org_micro_f1}."
                f"MISC_micro_precision: {misc_micro_precision}, MISC_micro_recall: {misc_micro_recall}, MISC_micro_f1: {misc_micro_f1}."
            )

            acc_epoch.append(micro_acc)
            precision_epoch.append(micro_precision)
            recall_epoch.append(micro_recall)
            f1_epoch.append(micro_f1)

            per_precision_epoch.append(micro_precision)
            per_recall_epoch.append(micro_recall)
            per_f1_epoch.append(micro_f1)
            
            loc_precision_epoch.append(micro_precision)
            loc_recall_epoch.append(micro_recall)
            loc_f1_epoch.append(micro_f1)
            
            org_precision_epoch.append(micro_precision)
            org_recall_epoch.append(micro_recall)
            org_f1_epoch.append(micro_f1)
            
            misc_precision_epoch.append(micro_precision)
            misc_recall_epoch.append(micro_recall)
            misc_f1_epoch.append(micro_f1)
            
            if micro_f1 >= max(f1_epoch):
                self.save_model(f'Best_model_from_{epoch + 1}')
                best_epoch = epoch + 1
                print(f"Best model saved ,epoch is {epoch + 1}")
        # 日志保存
        self.save_log(loss_epoch, acc_epoch, precision_epoch, recall_epoch, f1_epoch, best_epoch,
                      per_precision_epoch, per_recall_epoch, per_f1_epoch, 
                      loc_precision_epoch, loc_recall_epoch, loc_f1_epoch,
                      org_precision_epoch, org_recall_epoch, org_f1_epoch,
                      misc_precision_epoch, misc_recall_epoch, misc_f1_epoch)

    def test(self):
        self.model.eval()
        with torch.no_grad():
            test_bar = tqdm(self.test_dataloader, desc=f"Testing")
            correct_pre_position, total_token = 0, 0
            correct_ne, total_pred_entities, total_gold_entities = 0, 0, 0
            
            per_correct_ne = 0
            per_total_pred_entities = 0
            per_total_gold_entities = 0

            loc_correct_ne = 0
            loc_total_pred_entities = 0
            loc_total_gold_entities = 0

            org_correct_ne = 0
            org_total_pred_entities = 0
            org_total_gold_entities = 0

            misc_correct_ne = 0
            misc_total_pred_entities = 0
            misc_total_gold_entities = 0            
            
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
                    
                    per_correct_ne += metric['micro_metric'][5]
                    per_total_pred_entities += metric['micro_metric'][6]
                    per_total_gold_entities += metric['micro_metric'][7]
                        
                    loc_correct_ne += metric['micro_metric'][8]
                    loc_total_pred_entities += metric['micro_metric'][9]
                    loc_total_gold_entities += metric['micro_metric'][10]
                        
                    org_correct_ne += metric['micro_metric'][11]
                    org_total_pred_entities += metric['micro_metric'][12]
                    org_total_gold_entities += metric['micro_metric'][13]
                        
                    misc_correct_ne += metric['micro_metric'][14]
                    misc_total_pred_entities += metric['micro_metric'][15]
                    misc_total_gold_entities += metric['micro_metric'][16]
                    
                macro_acc += metric['macro_metric'][0]
                macro_precision += metric['macro_metric'][1]
                macro_recall += metric['macro_metric'][2]

            micro_acc = correct_pre_position / total_token
            micro_precision = correct_ne / total_pred_entities
            micro_recall = correct_ne / total_gold_entities
            micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall)
            
            per_micro_precision = per_correct_ne / per_total_pred_entities if per_total_pred_entities != 0 else 0.0
            per_micro_recall = per_correct_ne / per_total_gold_entities if per_total_gold_entities != 0 else 0.0 
            per_micro_f1 = 2 * (per_micro_precision * per_micro_recall) / (per_micro_precision + per_micro_recall) if per_micro_precision + per_micro_recall != 0 else 0.0 

            loc_micro_precision = loc_correct_ne / loc_total_pred_entities if loc_total_pred_entities != 0 else 0.0
            loc_micro_recall = loc_correct_ne / loc_total_gold_entities if loc_total_gold_entities != 0 else 0.0
            loc_micro_f1 = 2 * (loc_micro_precision * loc_micro_recall) / (loc_micro_precision + loc_micro_recall) if loc_micro_precision + loc_micro_recall != 0 else 0.0 

            
            org_micro_precision = org_correct_ne / org_total_pred_entities if org_total_pred_entities != 0 else 0.0
            org_micro_recall = org_correct_ne / org_total_gold_entities if org_total_gold_entities != 0 else 0.0
            org_micro_f1 = 2 * (org_micro_precision * org_micro_recall) / (org_micro_precision + org_micro_recall) if org_micro_precision + org_micro_recall != 0 else 0.0 

            
            misc_micro_precision = misc_correct_ne / misc_total_pred_entities if misc_total_pred_entities != 0 else 0.0
            misc_micro_recall = misc_correct_ne / misc_total_gold_entities if misc_total_gold_entities != 0 else 0.0 
            misc_micro_f1 = 2 * (misc_micro_precision * misc_micro_recall) / (misc_micro_precision + misc_micro_recall) if misc_micro_precision + misc_micro_recall != 0 else 0.0       

            macro_acc /= len(self.valid_dataloader)
            macro_precision /= len(self.valid_dataloader)
            macro_recall /= len(self.valid_dataloader)
            macro_f1 = 2 * (macro_precision * macro_recall) / (macro_precision + macro_recall)

            file_path = self.save_path + 'Test_result.txt'
            with open(file_path, 'w') as file:
                file.write(
                    f"micro_acc: {micro_acc}, micro_precision: {micro_precision}, micro_recall: {micro_recall}, micro_f1: {micro_f1}. "
                    f"macro_acc: {macro_acc}, macro_precision: {macro_precision}, macro_recall: {macro_recall}, macro_f1: {macro_f1}."
                    f"PER_micro_precision: {per_micro_precision}, PER_micro_recall: {per_micro_recall}, PER_micro_f1: {per_micro_f1}."
                    f"LOC_micro_precision: {loc_micro_precision}, LOC_micro_recall: {loc_micro_recall}, LOC_micro_f1: {loc_micro_f1}."
                    f"ORG_micro_precision: {org_micro_precision}, ORG_micro_recall: {org_micro_recall}, ORG_micro_f1: {org_micro_f1}."
                    f"MISC_micro_precision: {misc_micro_precision}, MISC_micro_recall: {misc_micro_recall}, MISC_micro_f1: {misc_micro_f1}."             
                )
            file.close()
            print("write to Test_result.txt successfully")

    def save_model(self, model_name):
        torch.save(self.model.state_dict(), self.save_path + model_name)
        print("Model save successfully")

    def save_log(self, loss_epoch, acc_epoch, precision_epoch, recall_epoch, f1_epoch, best_epoch, 
                      per_precision_epoch, per_recall_epoch, per_f1_epoch, 
                      loc_precision_epoch, loc_recall_epoch, loc_f1_epoch,
                      org_precision_epoch, org_recall_epoch, org_f1_epoch,
                      misc_precision_epoch, misc_recall_epoch, misc_f1_epoch):
        data = {
            'loss': loss_epoch,
            'acc': acc_epoch,
            'precision': precision_epoch,
            'recall': recall_epoch,
            'f1': f1_epoch,
            'PER_precision': per_precision_epoch,
            'PER_recall': per_recall_epoch,
            'PER_f1': per_f1_epoch,
            'LOC_precision': loc_precision_epoch,
            'LOC_recall': loc_recall_epoch,
            'LOC_f1': loc_f1_epoch,
            'ORG_precision': org_precision_epoch,
            'ORG_recall': org_recall_epoch,
            'ORG_f1': org_f1_epoch,
            'MISC_precision': misc_precision_epoch,
            'MISC_recall': misc_recall_epoch,
            'MISC_f1': misc_f1_epoch,
            'best_epoch': [best_epoch] + [None] * (len(loss_epoch) - 1)
        }
        df = pd.DataFrame(data)
        df.to_csv(self.save_path + 'Train_log.csv', index=False)
        print('Log save successfully')
