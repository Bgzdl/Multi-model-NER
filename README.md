# Multi-model-NER
Use multi-source domain information to complete NER tasks
# 数据准备
将数据集文件夹放置在与该项目同级的目录下
```
├─IJCAI2019_data
└─Multi-model-NER
```
# 模型参数准备
将模型参数放置在该项目的根目录下
```
├─distilbert-NER
│  └─...
│  .gitignore
│  README.md
│  train.py 
└─ ...

```
# 代码结构
```
│  .gitignore
│  README.md
│  train.py                               # 训练测试脚本
│
├─Data_Processing
│  └─  ReadData.py                        # 数据集定义
│
├─Fusion_Model
│  │  fusion_layer.py                     # 多模态信息融合层定义
│  └─ fusion_model.py                     # 融合模型定义
│          
├─Image_Encoder                           # 视觉特征提取器
│  │  build_image_encoder.py              # 图像特征提取器构建脚本
│  │  image_encoder.py                    # 图像特征提取器
│  │  LinearProbe.py                      # 线性投影层
│  │  __init__.py
│  │  
│  └─clip                                 # clip包
│     │  bpe_simple_vocab_16e6.txt.gz
│     │  clip.py
│     │  lora_model.py                    # 增加Lora层的clip视觉模型
│     │  model.py
│     │  simple_tokenizer.py
│     └─ __init__.py
│          
├─Pipline                                 # 命名实体识别分类头
│  │  pipline.py                          # 分类头定义
│  └─ __init__.py
│          
├─Text_Encoder                            # 文本特征提取器
│  │  build_text_encoder.py               # 文本特征提取器构建脚本
│  └─ __init__.py
│          
└─Trainer                                 # 训练器
    │  trainer.py                         # 训练器定义
    └─ __init__.py
```
# 运行说明
## 环境说明
pass
## 运行命令
```
# 切换到该目录下
python train.py
```