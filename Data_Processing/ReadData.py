import torch
from PIL import Image
import os

from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, filename):
        self.filename = filename
        self.image = []
        self.token = []
        self.label = []

        f = open(os.path.join(filename, 'twitter2015/train.txt'), encoding='utf-8')
        data = []
        imgs = []
        auxlabels = []
        sentence = []
        label= []
        auxlabel = []
        imgid = ''
        for line in f:
            if line.startswith('IMGID:'):
                imgid = line.strip().split('IMGID:')[1]+'.jpg'
                continue
            if line[0]=="\n":
                if len(sentence) > 0:
                    data.append((sentence,label))
                    imgs.append('twitter2015_images/'+str(imgid))
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

        if len(sentence) >0:
            data.append((sentence,label))
            imgs.append(imgid)
            auxlabels.append(auxlabel)
            sentence = []
            label = []
            auxlabel = []

        f.close()
        f = open(os.path.join(filename, 'twitter2017/train.txt'), encoding='utf-8')
        imgid = ''
        for line in f:
            if line.startswith('IMGID:'):
                imgid = line.strip().split('IMGID:')[1]+'.jpg'
                continue
            if line[0]=="\n":
                if len(sentence) > 0:
                    data.append((sentence,label))
                    imgs.append('twitter2017_images/'+str(imgid))
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

        if len(sentence) >0:
            data.append((sentence,label))
            imgs.append(imgid)
            auxlabels.append(auxlabel)
            sentence = []
            label = []
            auxlabel = []

        for i in range(len(data)):
            for j in range(len(data[i][0])):
                if(j>3 and j< len(data[i][0])-2):
                    self.token.append(data[i][0][j])
                    self.label.append(data[i][1][j])
                    self.image.append(imgs[i])

    def __len__(self):
        return len(self.token)

    def __getitem__(self, index):
        # 获取数据和标签
        image = os.path.join(self.filename, self.image[index])
        token = self.token[index]
        label = self.label[index]

        # 进行任何必要的处理，例如：数据增强、标准化等
        # data = ...

        return image, token, label

if __name__ == "__main__":
    A = CustomDataset('../IJCAI2019_data')
    print(A[0],A[2])