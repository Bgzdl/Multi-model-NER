import re

import torch

dict_label = {
    None: -1,
    
    'O': 0,
    
    'LOC': 1,

    'PER': 2,

    'MISC': 3,

    'ORG': 4,

}
inverse_dict_label = {v: k for k, v in dict_label.items()}


def clean_text(text):
    # 处理单个符号的情况
    if len(text) == 1 and not text.isalnum():
        return text
    # 处理混合符号和字母的情况
    cleaned_text = re.sub(r'[^a-zA-Z0-9]', '', text)

    return cleaned_text


def pad_list_to_length(lst, target_length, pad_value=None):
    # 计算需要添加的元素数量
    padding = [pad_value] * (target_length - len(lst))
    # 如果列表已经超过或等于目标长度，不进行填充
    return lst[:target_length] + padding if len(lst) < target_length else lst[:target_length]


def extract_token(token_idxs_list, tokenizer):
    tokenized_texts = []

    for token_idxs in token_idxs_list:
        token_idxs = [idx for idx in token_idxs if idx not in [0, 101, 102]]
        tokens = tokenizer.convert_ids_to_tokens(token_idxs)
        tokenized_texts.append(tokens)

    return tokenized_texts


def extract_named_entity(label_seq, tokenized_text):
    named_entity = {}
    entity = ''
    entity_type_count = {}

    if isinstance(label_seq, torch.Tensor):
        label_seq = label_seq.tolist()

    for i, (token, label) in enumerate(zip(tokenized_text, label_seq)):
        if label != 0:
            if token.startswith('##'):
                entity += token[2:]
                if label not in entity_type_count:
                    entity_type_count[label] = 1
                else:
                    entity_type_count[label] += 1
            else:
                if entity:
                    entity_type = max(entity_type_count, key=entity_type_count.get)
                    named_entity[entity] = inverse_dict_label[entity_type]
                # reset
                entity = token
                entity_type_count[label] = 1

    if entity:
        entity_type = max(entity_type_count, key=entity_type_count.get)
        named_entity[entity] = inverse_dict_label[entity_type]

    return named_entity


def calculate_ner_accuracy(gold_named_entity, pred_named_entity):
    correct_predictions = 0
    total_pred_entities = len(pred_named_entity)
    total_gold_entities = len(gold_named_entity)

    # 遍历预测的命名实体
    for word, predicted_entity in pred_named_entity.items():
        # 检查该单词是否在真实标注中且实体类型匹配
        if word in gold_named_entity and predicted_entity == gold_named_entity[word]:
            correct_predictions += 1

    return correct_predictions, total_pred_entities, total_gold_entities

def calculate_per_accuracy(gold_named_entity, pred_named_entity):
    correct_predictions = 0
    total_pred_entities = 0
    total_gold_entities = 0

    # 遍历预测的命名实体
    for word, predicted_entity in pred_named_entity.items():
        if predicted_entity == 'PER':
            total_pred_entities += 1
        # 检查该单词是否在真实标注中且实体类型匹配
            if word in gold_named_entity and predicted_entity == gold_named_entity[word]:
                correct_predictions += 1
    for word, gold_entity in gold_named_entity.items():
        if gold_entity == 'PER':
            total_gold_entities += 1
    return correct_predictions, total_pred_entities, total_gold_entities

def calculate_loc_accuracy(gold_named_entity, pred_named_entity):
    correct_predictions = 0
    total_pred_entities = 0
    total_gold_entities = 0

    # 遍历预测的命名实体
    for word, predicted_entity in pred_named_entity.items():
        if predicted_entity == 'LOC':
            total_pred_entities += 1
        # 检查该单词是否在真实标注中且实体类型匹配
            if word in gold_named_entity and predicted_entity == gold_named_entity[word]:
                correct_predictions += 1
    for word, gold_entity in gold_named_entity.items():
        if gold_entity == 'LOC':
            total_gold_entities += 1
    return correct_predictions, total_pred_entities, total_gold_entities

def calculate_org_accuracy(gold_named_entity, pred_named_entity):
    correct_predictions = 0
    total_pred_entities = 0
    total_gold_entities = 0

    # 遍历预测的命名实体
    for word, predicted_entity in pred_named_entity.items():
        if predicted_entity == 'ORG':
            total_pred_entities += 1
        # 检查该单词是否在真实标注中且实体类型匹配
            if word in gold_named_entity and predicted_entity == gold_named_entity[word]:
                correct_predictions += 1
    for word, gold_entity in gold_named_entity.items():
        if gold_entity == 'ORG':
            total_gold_entities += 1
    return correct_predictions, total_pred_entities, total_gold_entities

def calculate_misc_accuracy(gold_named_entity, pred_named_entity):
    correct_predictions = 0
    total_pred_entities = 0
    total_gold_entities = 0

    # 遍历预测的命名实体
    for word, predicted_entity in pred_named_entity.items():
        if predicted_entity == 'MISC':
            total_pred_entities += 1
        # 检查该单词是否在真实标注中且实体类型匹配
            if word in gold_named_entity and predicted_entity == gold_named_entity[word]:
                correct_predictions += 1
    for word, gold_entity in gold_named_entity.items():
        if gold_entity == 'MISC':
            total_gold_entities += 1
    return correct_predictions, total_pred_entities, total_gold_entities

if __name__ == '__main__':
    predicted_seq = [0,  0,  0,  0,  0,  0,  2,  2,  2,  2,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]
    tokenized_text = ['_', ':', 'Me', 'outside', 'of', 'where', 'George', 'Z', '##immer', '##man', 'got', 'shot', 'at', '.', 'You', 'know', 'God', 'is', 'so', 'good', '.']
    named_entity = extract_named_entity(predicted_seq, tokenized_text)
    print(named_entity)
