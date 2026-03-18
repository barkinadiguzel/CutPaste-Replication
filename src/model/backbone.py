import torch
import torch.nn as nn
import torchvision.models as models

def get_backbone(pretrained=True):
    backbone = models.resnet18(pretrained=pretrained)
    modules = list(backbone.children())[:-1] 
    backbone = nn.Sequential(*modules)
    return backbone
