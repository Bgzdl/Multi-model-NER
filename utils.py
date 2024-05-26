import re

dict_label = {
    None: -1,
    'O': 0,
    'B-LOC': 1,
    'I-LOC': 2,
    'B-PER': 3,
    'I-PER': 4,
    'B-MISC': 5,
    'I-MISC': 6,
    'B-ORG': 7,
    'I-ORG': 8,
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


def extract_named_entity(predicted_seq, tokenized_text):
    named_entity = {}
    entity = ''
    entity_type_count = {}

    for i, (token, label) in enumerate(zip(tokenized_text, predicted_seq)):
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


if __name__ == '__main__':
    predicted_seq = [0,  0,  0,  0,  0,  0,  3,  4,  4,  4,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]
    tokenized_text = ['_', ':', 'Me', 'outside', 'of', 'where', 'George', 'Z', '##immer', '##man', 'got', 'shot', 'at', '.', 'You', 'know', 'God', 'is', 'so', 'good', '.']
    named_entity = extract_named_entity(predicted_seq, tokenized_text)
    print(named_entity)
