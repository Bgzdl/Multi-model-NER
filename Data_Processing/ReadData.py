import os
import torch
import numpy as np
from PIL import Image
import PIL
from torch.utils.data import Dataset
from utils import *

wrong_jpg = ['O_2548.jpg', 'O_1478.jpg', 'O_2955.jpg', 'O_2366.jpg', 'O_2430.jpg', 'O_2590.jpg']


class CustomDataset(Dataset):
    @staticmethod
    def label_mapping(label):
        return [dict_label[label[i]] for i in range(len(label))]

    def load_data(self, data_path, file_name):
        f = open(os.path.join(data_path, file_name), encoding='utf-8')
        data = []
        data_img = []

        sentence = []
        label = []
        imgid = ''
        for line in f:
            if line.startswith('IMGID:'):
                imgid = line.strip().split('IMGID:')[1] + '.jpg'
                continue
            if line[0] == "\n":
                if len(sentence) > 0:
                    try:
                        with Image.open(os.path.join(data_path, file_name.split('/')[0] + '_images', imgid)) as im:
                            single_sentence = ' '.join(sentence[1:len(sentence)])
                            label = label[1:len(label)]
                            data.append((single_sentence, label))
                            prefix = file_name.split('/')[0] + '_images'
                            data_img.append(f'{prefix}/' + str(imgid))
                            sentence = []
                            label = []
                            imgid = ''
                    except IOError:
                        pass
                continue

            splits = line.split('\t')

            # clean up irrelevant words
            if splits[0].find('http') < 0 and splits[0].find('@') != 0:
                sentence.append(clean_text(splits[0]))

                cur_label = splits[-1][:-1]

                if cur_label == 'B-OTHER' or cur_label == 'I-OTHER' or cur_label == 'B-MISC' or cur_label == 'I-MISC':
                    cur_label = 'MISC'
                elif cur_label == 'B-ORG' or cur_label == 'I-ORG':
                    cur_label = 'ORG'
                elif cur_label == 'B-LOC' or cur_label == 'I-LOC':
                    cur_label = 'LOC'    
                elif cur_label == 'B-PER' or cur_label == 'I-PER':
                    cur_label = 'PER'

                label.append(cur_label)

        if len(sentence) > 0:
            try:
                with Image.open(os.path.join(data_path, file_name.split('/')[0] + '_images', imgid)) as im:
                    single_sentence = ' '.join(sentence[1:len(sentence)])
                    label = label[1:len(label)]
                    data.append((single_sentence, label))
                    prefix = file_name.split('/')[0] + '_images'
                    data_img.append(f'{prefix}/' + str(imgid))
            except IOError:
                pass
        f.close()
        return data, data_img

    def __init__(self, filename, mode, image_preprocess=None, text_preprocess=None, max_len=32):
        self.filename = filename
        self.image = []
        self.sentence = []
        self.label = []
        self.image_preprocess = image_preprocess
        self.text_preprocess = text_preprocess
        self.max_len = max_len
        # 读取数据
        if mode == 'train':
            data, imgs = self.load_data(filename, 'twitter2015/train.txt')
            data_1, imgs_1 = self.load_data(filename, 'twitter2017/train.txt')
        elif mode == 'valid':
            data, imgs = self.load_data(filename, 'twitter2015/valid.txt')
            data_1, imgs_1 = self.load_data(filename, 'twitter2017/valid.txt')
        elif mode == 'test':
            data, imgs = self.load_data(filename, 'twitter2015/test.txt')
            data_1, imgs_1 = self.load_data(filename, 'twitter2017/test.txt')
        else:
            raise Exception("Mode Error, please choose 'train', 'valid' or 'test'")
        # 合并数据
        data.extend(data_1)
        imgs.extend(imgs_1)

        for i in range(len(data)):
            self.sentence.append(data[i][0])
            if len(data[i][1]) == 0:
                # print(data[i][0], data[i][1])
                print(imgs[i])
            self.label.append(data[i][1])
            self.image.append(imgs[i])

        self.image = np.array(self.image)

    def __len__(self):
        return len(self.image)

    def __getitem__(self, index):
        # 获取数据和标签
        if len(self.sentence[index]) == 0:
            print(f'Sentence length is zero, image file name is {os.path.join(self.filename, self.image[index])}.')

        try:
            image = Image.open(os.path.join(self.filename, self.image[index]))
        except PIL.UnidentifiedImageError:
            print(f'Image error, image file name is {self.image[index]}.')
            image = Image.open(os.path.join('../IJCAI2019_data/twitter2017_images', '16_05_01_6.jpg'))

        # image preprocess
        if self.image_preprocess:
            image = self.image_preprocess(image)

        # text and label preprocess
        sentence = self.sentence[index]
        label = self.label[index]
        sentence_len = len(label)
        if self.text_preprocess:
            # label preprocess
            tokenized_text = self.text_preprocess.tokenize(sentence)
            sentence_len = len(tokenized_text) + 2  # actual valid len of sentence, include CLS and SEP token
            aligned_label = []
            valid_token = 0

            for i in range(len(tokenized_text)):
                # extend label for BERT tokenizer
                if tokenized_text[i].startswith("##"):
                    aligned_label.append(aligned_label[i - 1])
                else:
                    aligned_label.append(label[min(valid_token, sentence_len)])
                    valid_token += 1

            aligned_label.insert(0, 'O')  # pre-defined label of CLS token
            aligned_label.append('O')  # pre-defined label of SEP token
            aligned_label = pad_list_to_length(aligned_label, self.max_len, None)
            label = torch.tensor(self.label_mapping(aligned_label))

            # text preprocess
            sentence = self.text_preprocess(
                sentence,
                max_length=self.max_len,
                truncation=True,
                padding="max_length",
                return_tensors='pt'
            )

            for key, value in sentence.items():
                sentence[key] = torch.squeeze(value, dim=0)

        return image, sentence, label, sentence_len


if __name__ == "__main__":
    # training set
    A = CustomDataset('../IJCAI2019_data', 0)
    for i in range(7122):
        A[i]
    # validation set
    B = CustomDataset('../IJCAI2019_data', 1)
    for i in range(1664):
        B[i]
    # test set
    C = CustomDataset('../IJCAI2019_data', 2)
    for i in range(3929):
        C[i]
