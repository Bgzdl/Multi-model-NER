import clip
import torch.nn as nn
from clip.model import VisionTransformer
import torchvision.transforms.transforms as transforms
from clip.lora_model import add_Lora_to_visual_model


class LinearProbe(nn.Module):
    def __init__(self, output_dim: int):
        super(LinearProbe).__init__()
        self.projection = nn.Linear(output_dim, 512)

    def forward(self, x):
        x = self.projection(x)
        return x


def build_image_model(model_name) -> [VisionTransformer, transforms.Compose]:
    """
    Loading the Clip model parameters and return the Image encoder and preprocess
    :param model_name: one of ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64',
     'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']
    :return: Image encoder
    """
    model, preprocess = clip.load(model_name)
    projection = LinearProbe(model.visual.output_dim)
    visual_model = nn.Sequential(model.visual, projection)
    return visual_model, preprocess


def build_Lora_image_model(model_name, low_rank) -> [VisionTransformer, transforms.Compose]:
    """
        Loading the Clip model parameters and return the Image encoder and preprocess
        :param model_name: one of ['ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']
        :return: Image encoder
        """
    model, preprocess = clip.load(model_name)
    projection = LinearProbe(model.visual.output_dim)
    # 为模型添加lora层
    add_Lora_to_visual_model(model.visual, low_rank)
    visual_model = nn.Sequential(model.visual, projection)
    return visual_model, preprocess
