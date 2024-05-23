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

        data_img = []

        sentence = []
        label_O = []
        label_T = []

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

                            label_O = label_O[1:len(label_O)]
                            label_T = label_T[1:len(label_T)]

                            data.append((single_sentence, label_O, label_T))
                            prefix = file_name.split('/')[0] + '_images'
                            data_img.append(f'{prefix}/' + str(imgid))
                            sentence = []
                            label_O = []
                            label_T = []
                            imgid = ''

                    except IOError:
                        pass
                continue
            splits = line.split('\t')
            sentence.append(splits[0])
            cur_label = splits[-1][:-1]
            if cur_label == 'B-OTHER' or cur_label == 'I-OTHER' or cur_label == 'B-MISC' or cur_label == 'I-MISC':
                cur_label = 'MISC'
            elif cur_label == 'B-ORG' or cur_label == 'I-ORG':
                cur_label = 'ORG'
            elif cur_label == 'B-LOC' or cur_label == 'I-LOC':
                cur_label = 'LOC'    
            elif cur_label == 'B-PER' or cur_label == 'I-PER':
                cur_label = 'PER'
            if cur_label == 'O':
                label_O.append(0)
            else:
                label_O.append(1)
            label_T.append(cur_label)


        if len(sentence) > 0:
            try:
                with Image.open(os.path.join(data_path, file_name.split('/')[0] + '_images', imgid)) as im:
                    single_sentence = ' '.join(sentence[1:len(sentence)])
                    data_sentence.append(single_sentence)

                    label_O = label_O[1:len(label_O)]
                    label_T = label_T[1:len(label_T)]

                    data.append((single_sentence, label_O, label_T))
                    prefix = file_name.split('/')[0] + '_images'
                    data_img.append(f'{prefix}/' + str(imgid))
                    sentence = []
                    label_O = []
                    label_T = []
                    imgid = ''

            except IOError:
                pass
        f.close()
        return data, data_img

    def __init__(self, filename, mode, image_preprocess=None, text_preprocess=None, max_len=32):
        self.filename = filename
        self.image = []
        self.sentence = []
        # 标签是否为O（O为0，非O为1）
        self.label_O = []
        # 标签真实值
        self.label_T = []
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
            self.label_O.append(CustomDataset.pad_list_to_length(data[i][1], max_sentence_len, None))
            self.label_T.append(CustomDataset.pad_list_to_length(data[i][2], max_sentence_len, None))
            self.image.append(imgs[i])

        # image和token类型转换
        self.image = np.array(self.image)

        dictlabel_T = {'PER': [0, 0, 0, 0, 1], 'LOC': [0, 0, 0, 1, 0], 'ORG': [0, 0, 1, 0, 0],
                     'MISC': [0, 1, 0, 0, 0], 'O': [1, 0, 0, 0, 0], None: [0, 0, 0, 0, 0]}
        for i in range(len(self.label_T)):
            for j in range(len(self.label_T[i])):
                self.label_T[i][j] = torch.tensor(dictlabel_T[self.label_T[i][j]])
        self.label_T = [torch.stack(item) for item in self.label_T]

        dictlabel_O = {0: [0, 1], 1: [1, 0], None: [0, 0]}
        for i in range(len(self.label_O)):
            for j in range(len(self.label_O[i])):
                self.label_O[i][j] = torch.tensor(dictlabel_O[self.label_O[i][j]])
        self.label_O = [torch.stack(item) for item in self.label_O]

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

        label_O = self.label_O[index]
        label_T = self.label_T[index]
        sentence_len = self.sentence_len[index]
        return image, sentence, label_O, label_T, sentence_len



if __name__ == "__main__":
    # training set
    A = CustomDataset('E:/NLP/IJCAI2019_data', "train")
    for i in range(7122):
        A[i]
    # validation set
    B = CustomDataset('E:/NLP/IJCAI2019_data', "valid")
    for i in range(1664):
        B[i]
    # test set
    C = CustomDataset('E:/NLP/IJCAI2019_data', "test")
    for i in range(3929):
        C[i]
