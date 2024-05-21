import os
import torch
import numpy as np
from PIL import Image
import PIL
from torch.utils.data import Dataset

wrong_jpg = ['O_2548.jpg', 'O_1478.jpg', 'O_2955.jpg', 'O_2366.jpg', 'O_2430.jpg', 'O_2590.jpg']


class CustomDataset(Dataset):
    @staticmethod
    def pad_list_to_length(lst, target_length, pad_value=None):
        # 计算需要添加的元素数量
        padding = [pad_value] * (target_length - len(lst))
        # 如果列表已经超过或等于目标长度，不进行填充
        return lst[:target_length] + padding if len(lst) < target_length else lst[:target_length]

    def load_data(self, data_path, file_name):
        f = open(os.path.join(data_path, file_name), encoding='utf-8')
        data = []
        data_sentence = []
        data_label = []
        data_img = []
        data_auxlabel = []

        sentence = []
        label = []
        auxlabel = []
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
                            data_sentence.append(single_sentence)
                            label = label[1:len(label)]
                            data_label.append(label)
                            data.append((single_sentence, label))
                            prefix = file_name.split('/')[0] + '_images'
                            data_img.append(f'{prefix}/' + str(imgid))
                            data_auxlabel.append(auxlabel)
                            sentence = []
                            label = []
                            imgid = ''
                            auxlabel = []
                    except IOError:
                        pass
                continue
            splits = line.split('\t')
            sentence.append(splits[0])
            cur_label = splits[-1][:-1]
            if cur_label == 'B-OTHER':
                cur_label = 'B-MISC'
            elif cur_label == 'I-OTHER':
                cur_label = 'I-MISC'
            label.append(cur_label)
            auxlabel.append(cur_label[0])

        if len(sentence) > 0:
            try:
                with Image.open(os.path.join(data_path, file_name.split('/')[0] + '_images', imgid)) as im:
                    single_sentence = ' '.join(sentence[1:len(sentence)])
                    data_sentence.append(single_sentence)
                    label = label[1:len(label)]
                    data_label.append(label)
                    data.append((single_sentence, label))
                    prefix = file_name.split('/')[0] + '_images'
                    data_img.append(f'{prefix}/' + str(imgid))
                    data_auxlabel.append(auxlabel)
            except IOError:
                pass
        f.close()
        return data, data_img

    def __init__(self, filename, mode, image_preprocess=None, text_preprocess=None, max_len=32):
        self.filename = filename
        self.image = []
        self.sentence = []
        self.label = []
        self.sentence_len = []
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
        max_sentence_len = 0
        for item in data:
            max_sentence_len = max(len(item[1]), max_sentence_len)

        for i in range(len(data)):
            self.sentence.append(data[i][0])
            self.sentence_len.append(len(data[i][1]))
            if len(data[i][1]) == 0:
                # print(data[i][0], data[i][1])
                print(imgs[i])
            self.label.append(CustomDataset.pad_list_to_length(data[i][1], max_sentence_len, None))
            self.image.append(imgs[i])

        # image和token类型转换
        self.image = np.array(self.image)

        dictlabel = {'B-ORG': [0, 0, 0, 0, 0, 0, 0, 0, 1], 'B-MISC': [0, 0, 0, 0, 0, 0, 0, 1, 0], 'I-ORG': [0, 0, 0, 0, 0, 0, 1, 0, 0],
                     'B-PER': [0, 0, 0, 0, 0, 1, 0, 0, 0], 'I-MISC': [0, 0, 0, 0, 1, 0, 0, 0, 0], 'O': [0, 0, 0, 1, 0, 0, 0, 0, 0],
                     'I-PER': [0, 0, 1, 0, 0, 0, 0, 0, 0], 'I-LOC': [0, 1, 0, 0, 0, 0, 0, 0, 0], 'B-LOC': [1, 0, 0, 0, 0, 0, 0, 0, 0],
                     None: [0, 0, 0, 0, 0, 0, 0, 0, 0]}
        for i in range(len(self.label)):
            for j in range(len(self.label[i])):
                self.label[i][j] = torch.tensor(dictlabel[self.label[i][j]])
        self.label = [torch.stack(item) for item in self.label]

        # self.image = self.image[:16]
        # self.label = self.label[:16]
        # self.sentence = self.sentence[:16]
        # self.sentence_len = self.sentence_len[:16]

    def __len__(self):
        return len(self.image)

    def __getitem__(self, index):
        # 获取数据和标签
        if self.sentence_len[index] == 0:
            print(f'Sentence length is zero, image file name is {os.path.join(self.filename, self.image[index])}.')
        try:
            image = Image.open(os.path.join(self.filename, self.image[index]))
        except PIL.UnidentifiedImageError:
            print(f'Image error, image file name is {self.image[index]}.')
            image = Image.open(os.path.join('../IJCAI2019_data/twitter2017_images', '16_05_01_6.jpg'))
        if self.image_preprocess:
            image = self.image_preprocess(image)
        sentence = self.sentence[index]

        if self.text_preprocess:
            sentence = self.text_preprocess(sentence, max_length=self.max_len, truncation=True, padding="max_length", return_tensors='pt')
            for key, value in sentence.items():
                sentence[key] = torch.squeeze(value, dim=0)

        label = self.label[index]
        sentence_len = self.sentence_len[index]
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
