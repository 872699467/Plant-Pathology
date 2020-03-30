from torchvision.models import resnet18
import torch
import torch.nn as nn
import os

os.environ['TORCH_HOME'] = ''  # 设置resnet18的预训练权重，默认为torch_home目录下


class PlantModel(nn.Module):

    def __init__(self, pretrained, num_class=4):
        super(PlantModel, self).__init__()
        self.backbone = resnet18(pretrained=pretrained)
        in_feature = self.backbone.fc.in_features
        self.logit = nn.Linear(in_feature, num_class)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.backbone.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)

        out = self.logit(x)
        return out


def save_checkpoints(state, is_best, fpath, fname):
    torch.save(state, os.path.join(fpath, fname))
    if is_best:
        torch.save(state, os.path.join(fpath, 'best_mode.pth'))
