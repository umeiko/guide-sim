
import numpy as np
# import collections
# import random
import torch
import torch.nn as nn
import torch.nn.functional as F
# import torchvision
from torchvision import models
from vit_pytorch import ViT

def replace_bn_with_identity(module:nn.Module):
    # 遍历当前模块的所有子模块
    for name, child in module.named_children(): 
        # 如果是 BN 层，则替换为 Identity 
        if isinstance(child, nn.BatchNorm2d):
            setattr(module, name, nn.Identity())
        else:
            # 对非 BN 子模块递归调用
            replace_bn_with_identity(child)

def initialize_weights(module:nn.modules):
    # 遍历当前模块的所有子模块
        for name, child in module.named_children(): 
            # 如果是 BN 层，则替换为 Identity 
            if isinstance(child, nn.Conv2d) or isinstance(child, nn.Linear):
                # nn.init.orthogonal_(module.weight, nn.init.calculate_gain('relu'))
                nn.init.xavier_uniform_(child.weight)
                # nn.init.kaiming_uniform_(module.weight)
                if child.bias is not None:
                    nn.init.constant_(child.bias, 0)
            else:
                replace_bn_with_identity(child)

class Hybrid_RESNET18_VIT3_FC(nn.Module):
    def __init__(self, input_channels=1, act_num=5, use_softmax=True):
        super().__init__()
        self.input_shape = (input_channels, 256, 256)
        # resnet18构造
        self.resnet18 = models.resnet18(weights=None) 
        replace_bn_with_identity(self.resnet18)
        self.resnet18.conv1  = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet18.fc  = nn.Identity()  # 使用 Identity 替代 fc 层
        self.use_softmax = use_softmax
        # VIT 构造
        self.vit = ViT(
                image_size = 256,
                channels = input_channels,
                patch_size = 32,
                num_classes = 5,
                dim = 1024,
                depth = 3,
                heads = 16,
                mlp_dim = 2048,
                dropout = 0.1,
                emb_dropout = 0.1
            )
        self.vit.mlp_head = nn.Linear(1024, 512)
        self.use_softmax = use_softmax
        
        self.hybrid_linear = nn.Linear(1024, 1024)
        
        self.critic_linear = nn.Linear(1024, 1)
        self.actor_linear = nn.Linear(1024, act_num)
        initialize_weights(self)

    def forward(self, x:torch.Tensor):
        for k, shape in enumerate(self.input_shape):
            if x.shape[k+1] != shape:
                raise ValueError(f"Input shape should be {self.input_shape}, got {x.shape[1:]}")
        # [b, 512]
        x1 = self.resnet18(x)
        # [b, 512]
        x2 = self.vit(x)
        # [b, 1024]
        x = torch.cat([x1, x2], dim=1)
        x = F.relu(self.hybrid_linear(x))

        if self.use_softmax:
            a = F.softmax(self.actor_linear(x), dim=1)
        else:
            a = self.actor_linear(x)
        return a, self.critic_linear(x)
    

class HYBRID_RESNET18_VITS_FC(nn.Module):
    def __init__(self, input_channels=1, act_num=5, use_softmax=True):
        super().__init__()
        self.input_shape = (input_channels, 256, 256)
        # resnet18构造
        self.resnet18 = models.resnet18(weights=None) 
        replace_bn_with_identity(self.resnet18)
        self.resnet18.conv1  = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet18.fc  = nn.Identity()  # 使用 Identity 替代 fc 层
        self.use_softmax = use_softmax
        # VIT 构造
        self.vit = ViT(
                image_size = 256,
                channels = input_channels,
                patch_size = 16,
                num_classes = 5,
                dim = 768,
                depth = 12,
                heads = 12,
                mlp_dim = 3072,
                dropout = 0.1,
                emb_dropout = 0.1
            )
        self.vit.mlp_head = nn.Linear(768, 512)
        self.use_softmax = use_softmax
        
        self.hybrid_linear = nn.Linear(1024, 1024)
        
        self.critic_linear = nn.Linear(1024, 1)
        self.actor_linear = nn.Linear(1024, act_num)
        initialize_weights(self)

    def forward(self, x:torch.Tensor):
        for k, shape in enumerate(self.input_shape):
            if x.shape[k+1] != shape:
                raise ValueError(f"Input shape should be {self.input_shape}, got {x.shape[1:]}")
        # [b, 512]
        x1 = self.resnet18(x)
        # [b, 512]
        x2 = self.vit(x)
        # [b, 1024]
        x = torch.cat([x1, x2], dim=1)
        x = F.gelu(self.hybrid_linear(x))

        if self.use_softmax:
            a = F.softmax(self.actor_linear(x), dim=1)
        else:
            a = self.actor_linear(x)
        return a, self.critic_linear(x)

class HYBRID_RESNET18_VIT24_FC(nn.Module):
    def __init__(self, input_channels=1, act_num=5, use_softmax=True):
        super().__init__()
        self.input_shape = (input_channels, 256, 256)
        # resnet18构造
        self.resnet18 = models.resnet18(weights=None) 
        replace_bn_with_identity(self.resnet18)
        self.resnet18.conv1  = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet18.fc  = nn.Identity()  # 使用 Identity 替代 fc 层
        self.use_softmax = use_softmax
        # VIT 构造
        self.vit = ViT(
                image_size = 256,
                channels = input_channels,
                patch_size = 32,
                num_classes = 5,
                dim = 1024,
                depth = 24,
                heads = 16,
                mlp_dim = 2048,
                dropout = 0.1,
                emb_dropout = 0.1
            )
        self.vit.mlp_head = nn.Linear(1024, 512)
        self.use_softmax = use_softmax
        
        self.hybrid_linear = nn.Linear(1024, 1024)
        
        self.critic_linear = nn.Linear(1024, 1)
        self.actor_linear = nn.Linear(1024, act_num)
        initialize_weights(self)

    def forward(self, x:torch.Tensor):
        for k, shape in enumerate(self.input_shape):
            if x.shape[k+1] != shape:
                raise ValueError(f"Input shape should be {self.input_shape}, got {x.shape[1:]}")
        # [b, 512]
        x1 = self.resnet18(x)
        # [b, 512]
        x2 = self.vit(x)
        # [b, 1024]
        x = torch.cat([x1, x2], dim=1)
        x = F.relu(self.hybrid_linear(x))

        if self.use_softmax:
            a = F.softmax(self.actor_linear(x), dim=1)
        else:
            a = self.actor_linear(x)
        return a, self.critic_linear(x)

class HYBRID_RESNET18_VIT4_FC(nn.Module):
    def __init__(self, input_channels=2, act_num=5, use_softmax=True):
        super().__init__()
        self.input_shape = (input_channels, 256, 256)
        # resnet18构造
        self.resnet18 = models.resnet18(weights=None) 
        replace_bn_with_identity(self.resnet18)
        self.resnet18.conv1  = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet18.fc  = nn.Identity()  # 使用 Identity 替代 fc 层
        self.use_softmax = use_softmax
        # VIT 构造
        self.vit = ViT(
                image_size = 256,
                channels = input_channels,
                patch_size = 32,
                num_classes = 5,
                dim = 1024,
                depth = 4,
                heads = 16,
                mlp_dim = 2048,
                dropout = 0.1,
                emb_dropout = 0.1
            )
        self.vit.mlp_head = nn.Linear(1024, 512)
        self.use_softmax = use_softmax
        
        self.hybrid_linear = nn.Linear(1024, 1024)
        
        self.critic_linear = nn.Linear(1024, 1)
        self.actor_linear = nn.Linear(1024, act_num)
        initialize_weights(self)

    def forward(self, x:torch.Tensor):
        for k, shape in enumerate(self.input_shape):
            if x.shape[k+1] != shape:
                raise ValueError(f"Input shape should be {self.input_shape}, got {x.shape[1:]}")
        # [b, 512]
        x1 = self.resnet18(x)
        # [b, 512]
        x2 = self.vit(x)
        # [b, 1024]
        x = torch.cat([x1, x2], dim=1)
        x = F.relu(self.hybrid_linear(x))

        if self.use_softmax:
            a = F.softmax(self.actor_linear(x), dim=1)
        else:
            a = self.actor_linear(x)
        return a, self.critic_linear(x)

MODEL_MAPPING = {
    "HYBRID_RESNET18_VITS_FC": HYBRID_RESNET18_VITS_FC,
    "HYBRID_RESNET18_VIT4": HYBRID_RESNET18_VIT4_FC,
}