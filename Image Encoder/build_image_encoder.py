import clip
import torch.nn as nn
import torch
from clip.model import VisionTransformer
import torchvision.transforms.transforms as transforms
from clip.lora_model import add_Lora_to_visual_model
from LinearProbe import LinearProbe


def build_image_model(model_name, froze: bool = True) -> [VisionTransformer, transforms.Compose]:
    """
    Loading the Clip model parameters and return the Image encoder and preprocess
    :param model_name: one of ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64',
     'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']
    :return: Image encoder
    """
    model, preprocess = clip.load(model_name)
    # To freeze the parameters in origin model
    if froze:
        for parameter in model.parameters():
            parameter.requires_grad = False
    # projection layer can be trained
    projection = LinearProbe(model.visual.output_dim, activate_fun=False)
    visual_model = nn.Sequential(model.visual, projection)
    return visual_model, preprocess


def build_Lora_image_model(model_name, low_rank=8, froze: bool = True) -> [VisionTransformer, transforms.Compose]:
    """
        Loading the Clip model parameters and return the Image encoder and preprocess
        :param model_name: one of ['ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']
        :return: Image encoder
        """
    model, preprocess = clip.load(model_name)
    # To freeze the parameters in origin model
    if froze:
        for parameter in model.parameters():
            parameter.requires_grad = False
    # projection layer can be trained
    projection = LinearProbe(model.visual.output_dim)
    # Adding lora to the model
    add_Lora_to_visual_model(model.visual, low_rank)

    visual_model = nn.Sequential(model.visual, projection)
    return visual_model, preprocess


if __name__ == "__main__":
    model, preprocess = build_Lora_image_model('ViT-B/32')
    print(model)
