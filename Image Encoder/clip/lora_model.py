import torch
import math
import torch.nn as nn
from .model import VisionTransformer, ResidualAttentionBlock


class LoRALayer:
    def __init__(
            self,
            r: int,
            lora_alpha: int,
            lora_dropout: float,
            merge_weights: bool,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights


class LoRA(nn.Module, LoRALayer):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            r: int = 8,
            lora_alpha: int = 1,
            lora_dropout: float = 0.0,
            # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
            merge_weights: bool = True,
    ):
        nn.Module.__init__(self)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)
        self.in_features = in_features
        self.out_features = out_features
        self.scaling = self.lora_alpha / self.r
        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        nn.init.zeros_(self.lora_B)

    def __repr__(self):
        return f'{self.__class__.__name__}(in_features={self.in_features}, out_features={self.out_features}, rank={self.r})'

    def forward(self, x):
        result = self.lora_dropout(x) @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)
        return result


class LoraResidualAttentionBlock(nn.Module):
    def __init__(self, origin_model: nn.Module, d_model: int, r: int):
        super().__init__()
        self.origin_model = origin_model
        for param in self.origin_model.parameters():
            param.requires_grad = False
        self.LoRA = LoRA(d_model, d_model, r)

    def forward(self, x: torch.Tensor):
        lora_features = self.LoRA(x)
        x = x + self.origin_model.attention(self.origin_model.ln_1(x))
        x = x + self.origin_model.mlp(self.origin_model.ln_2(x))
        x = lora_features + x
        return x


def add_Lora_to_visual_model(visual_model: VisionTransformer, low_rank: int):
    blocks = visual_model.transformer.resblocks
    new_blocks = []
    for block in blocks:
        new_blocks.append(LoraResidualAttentionBlock(block, block.d_model, low_rank))
    visual_model.transformer.resblocks = nn.Sequential(*new_blocks)
    # return visual_model
