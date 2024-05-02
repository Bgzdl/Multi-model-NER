import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from sklearn.preprocessing import OneHotEncoder


class CustomDataset(Dataset):
    def load_data(self, data_path, file_name):
        f = open(os.path.join(data_path, file_name), encoding='utf-8')
        data = []
        imgs = []
        auxlabels = []
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
                    data.append((sentence, label))
                    imgs.append('twitter2015_images/' + str(imgid))
                    auxlabels.append(auxlabel)
                    sentence = []
                    label = []
                    imgid = ''
                    auxlabel = []
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
            data.append((sentence, label))
            imgs.append('twitter2015_images/' + str(imgid))
            auxlabels.append(auxlabel)
        f.close()
        return data, imgs

    def __init__(self, filename, image_preprocess=None, text_preprocess=None, max_len=32):
        self.filename = filename
        self.image = []
        self.token = []
        self.label = []
        self.image_preprocess = image_preprocess
        self.text_preprocess = text_preprocess
        self.max_len = max_len
        # 读取数据
        data, imgs = self.load_data(filename, 'twitter2015/train.txt')
        data_1, imgs_1 = self.load_data(filename, 'twitter2017/train.txt')
        # 合并数据
        data.extend(data_1)
        imgs.extend(imgs_1)

        for i in range(len(data)):
            for j in range(len(data[i][0])):
                if (j > 3 and j < len(data[i][0]) - 2):
                    self.token.append(data[i][0][j])
                    self.label.append(data[i][1][j])
                    self.image.append(imgs[i])
        # image和token类型转换
        self.image = np.array(self.image)
        # label onehot 编码
        encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
        # 将列表转换为NumPy数组，并改变形状以符合OneHotEncoder的要求
        category_array = np.array(self.label).reshape(-1, 1)
        # 拟合并转换数据
        self.label = encoder.fit_transform(category_array)

    def __len__(self):
        return len(self.token)

    def __getitem__(self, index):
        # 获取数据和标签
        image = Image.open(os.path.join(self.filename, self.image[index]))
        if self.image_preprocess:
            image = self.image_preprocess(image)
        token = self.token[index]
        if self.text_preprocess:
            token = self.text_preprocess(token, max_length=self.max_len, truncation=True, padding="max_length", return_tensors='pt')
            for key, value in token.items():
                token[key] = torch.squeeze(value, dim=0)
        label = self.label[index]

        return image, token, label


if __name__ == "__main__":
    A = CustomDataset('../IJCAI2019_data')
    print(A[0], A[2])
