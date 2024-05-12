import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class CustomDataset(Dataset):
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
                        with Image.open(os.path.join(data_path, file_name.split('/')[0]+'_images', imgid)) as im:
                            single_sentence = ' '.join(sentence[1:len(sentence)])
                            data_sentence.append(single_sentence)
                            label = label[1:len(label)]
                            data_label.append(label)
                            data.append((single_sentence,label))
                            prefix = file_name.split('/')[0]+'_images'
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
                with Image.open(os.path.join(data_path, file_name.split('/')[0]+'_images', imgid)) as im:
                    single_sentence = ' '.join(sentence[1:len(sentence)])
                    data_sentence.append(single_sentence)
                    label = label[1:len(label)]
                    data_label.append(label)
                    data.append((single_sentence,label))
                    prefix = file_name.split('/')[0]+'_images'
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
        self.image_preprocess = image_preprocess
        self.text_preprocess = text_preprocess
        self.max_len = max_len
        # 读取数据
        if mode==0:
            data, imgs = self.load_data(filename, 'twitter2015/train.txt')
            data_1, imgs_1 = self.load_data(filename, 'twitter2017/train.txt')
        elif mode==1:
            data, imgs = self.load_data(filename, 'twitter2015/valid.txt')
            data_1, imgs_1 = self.load_data(filename, 'twitter2017/valid.txt')
        elif mode==2:
            data, imgs = self.load_data(filename, 'twitter2015/test.txt')
            data_1, imgs_1 = self.load_data(filename, 'twitter2017/test.txt')
        # 合并数据
        data.extend(data_1)
        imgs.extend(imgs_1)

        for i in range(len(data)):
            self.sentence.append(data[i][0])
            self.label.append(data[i][1])
            self.image.append(imgs[i])
        # image和token类型转换
        self.image = np.array(self.image)

        setlabel = []
        for i in self.label:
            for j in i:
                setlabel.append(j)
        setlabel = set(setlabel)

        dictlabel = {'B-ORG':[0,0,0,0,0,0,0,0,1], 'B-MISC':[0,0,0,0,0,0,0,1,0], 'I-ORG':[0,0,0,0,0,0,1,0,0], 
                     'B-PER':[0,0,0,0,0,1,0,0,0], 'I-MISC':[0,0,0,0,1,0,0,0,0], 'O':[0,0,0,1,0,0,0,0,0], 
                     'I-PER':[0,0,1,0,0,0,0,0,0], 'I-LOC':[0,1,0,0,0,0,0,0,0], 'B-LOC':[1,0,0,0,0,0,0,0,0]}
        for i in range(len(self.label)):
            for j in range(len(self.label[i])):
                self.label[i][j] = torch.tensor(dictlabel[self.label[i][j]])
        
    def __len__(self):
        return len(self.token)

    def __getitem__(self, index):
        # 获取数据和标签
        image = Image.open(os.path.join(self.filename, self.image[index]))
        if self.image_preprocess:
            image = self.image_preprocess(image)
        sentence = self.sentence[index]

        # setoflen = []
        # for i in range(len(self.sentence)):
        #     setoflen.append(len(self.sentence[i]))
        # setoflen = set(setoflen)
        # print(setoflen)

        if self.text_preprocess:
            sentence = self.text_preprocess(sentence, max_length=self.max_len, truncation=True, padding="max_length", return_tensors='pt')
            for key, value in sentence.items():
                sentence[key] = torch.squeeze(value, dim=0)

        label = self.label[index]
        # print(len(self.label))
        return image, sentence, label


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