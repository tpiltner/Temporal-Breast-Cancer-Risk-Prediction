#imageEncoder.py

#Image encoder 
#Uses a ResNet-18 backbone adapted to mammograms


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights


class image_Encoder(nn.Module):
    """
    image encoder (ResNet-18):
      - 1-channel conv1 adapted from ImageNet weights
      - returns last conv feature map by default: [B, 512, h, w]
      - can also return pooled embedding: [B, 512]
    """
    def __init__(self, pretrained: bool = True):
        super().__init__()

        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet18(weights=weights)

        # Adapt conv1: 3ch -> 1ch (average RGB filters)
        old_conv = backbone.conv1
        new_conv = nn.Conv2d(
            in_channels=1,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )
        if pretrained:
            with torch.no_grad():
                new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
        backbone.conv1 = new_conv

        # Keep layers up to layer4 (drop classifier head) 
        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.out_dim = backbone.fc.in_features  # 512 for ResNet-18

    def freeze(self):
        #Freeze backbone parameters 
        for p in self.parameters():
            p.requires_grad = False

        self.eval()
        return self
    
    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
        return self

    def forward(self, x: torch.Tensor, return_map: bool = True) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)  # [B,512,h,w]

        if return_map:
            return x

        x = F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)  # [B,512]
        return x


def get_image_encoder(pretrained: bool = True) -> image_Encoder:
    return image_Encoder(pretrained=pretrained)