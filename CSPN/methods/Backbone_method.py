import methods.wideresnet as wideresnet
import torch
import torch.nn as nn
from methods.resnet import ResNet
from torchvision import models as torchvision_models

class PretrainedResNet(nn.Module):

    def __init__(self, rawname, pretrain_path=None) -> None:
        super().__init__()
        if pretrain_path == 'default':
            self.model = torchvision_models.__dict__[rawname](pretrained=True)
            self.output_dim = self.model.fc.weight.shape[1]
            self.model.fc = nn.Identity()
        else:
            self.model = torchvision_models.__dict__[rawname]()
            self.output_dim = self.model.fc.weight.shape[1]
            self.model.fc = nn.Identity()
            if pretrain_path is not None:
                sd = torch.load(pretrain_path)
                self.model.load_state_dict(sd, strict=True)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        return x


class Backbone(nn.Module):

    def __init__(self, config, inchan):
        super().__init__()

        if config['backbone'] == 'wideresnet28-2':
            self.backbone = wideresnet.WideResNetBackbone(None, 28, 2, 0, config['cspn_model']['projection_dim'])
        elif config['backbone'] == 'wideresnet40-4':
            self.backbone = wideresnet.WideResNetBackbone(None, 40, 4, 0, config['cspn_model']['projection_dim'])
        elif config['backbone'] == 'wideresnet16-8':
            self.backbone = wideresnet.WideResNetBackbone(None, 16, 8, 0.4, config['cspn_model']['projection_dim'])
        elif config['backbone'] == 'wideresnet28-10':
            self.backbone = wideresnet.WideResNetBackbone(None, 28, 10, 0.3, config['cspn_model']['projection_dim'])
        elif config['backbone'] == 'resnet18':
            self.backbone = ResNet(output_dim=config['cspn_model']['projection_dim'], inchan=inchan)
        elif config['backbone'] == 'resnet18a':
            self.backbone = ResNet(output_dim=config['cspn_model']['projection_dim'], resfirststride=2,
                                   inchan=inchan)
        elif config['backbone'] == 'resnet18b':
            self.backbone = ResNet(output_dim=config['cspn_model']['projection_dim'], resfirststride=2,
                                   inchan=inchan)
        elif config['backbone'] == 'resnet34':
            self.backbone = ResNet(output_dim=config['cspn_model']['projection_dim'], num_block=[3, 4, 6, 3],
                                   inchan=inchan)
        elif config['backbone'] in ['prt_r18', 'prt_r34', 'prt_r50']:
            self.backbone = PretrainedResNet(
                {'prt_r18': 'resnet18', 'prt_r34': 'resnet34', 'prt_r50': 'resnet50'}[config['backbone']])
        elif config['backbone'] in ['prt_pytorchr18', 'prt_pytorchr34', 'prt_pytorchr50']:
            name, path = {
                'prt_pytorchr18': ('resnet18', 'default'),
                'prt_pytorchr34': ('resnet34', 'default'),
                'prt_pytorchr50': ('resnet50', 'default')
            }[config['backbone']]
            self.backbone = PretrainedResNet(name, path)
        elif config['backbone'] in ['prt_dinor18', 'prt_dinor34', 'prt_dinor50']:
            name, path = {
                'prt_dinor50': ('resnet50', './model_weights/dino_resnet50_pretrain.pth')
            }[config['backbone']]
            self.backbone = PretrainedResNet(name, path)
        else:
            bkb = config['backbone']
            raise Exception(f'Backbone \"{bkb}\" is not defined.')
        self.output_dim = self.backbone.output_dim

    def forward(self, x):
        x = self.backbone(x)
        return x