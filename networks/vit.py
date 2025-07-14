import torch
import torch.nn as nn
import torch.nn.functional as F
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

class VIT3_FC(nn.Module):
    def __init__(self, input_channels=1, act_num=5, use_softmax=True):
        super().__init__()
        self.input_shape = (input_channels, 256, 256)
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
        # replace_bn_with_identity(self.resnet)
        self.vit.mlp_head = nn.Identity()  
        self.use_softmax = use_softmax
        self.cov_out = nn.Sequential(
            nn.Linear(1024, 1024),
        )
        self.critic_linear = nn.Linear(1024, 1)
        self.actor_linear = nn.Linear(1024, act_num)
        initialize_weights(self)

    def forward(self, x:torch.Tensor):
        for k, shape in enumerate(self.input_shape):
            if x.shape[k+1] != shape:
                raise ValueError(f"Input shape should be {self.input_shape}, got {x.shape[1:]}")
        x = self.vit(x)
        x = F.relu(self.cov_out(x))
        if self.use_softmax:
            a = F.softmax(self.actor_linear(x), dim=1)
        else:
            a = self.actor_linear(x)
        return a, self.critic_linear(x)