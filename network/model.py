import torch
import torchvision
import torch.nn as nn
from .gem_pool import GeneralizedMeanPoolingP

class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('InstanceNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)

class RGB_Model(nn.Module):
    def __init__(self, pretrain_path="default"):
        super(RGB_Model, self).__init__()
        if pretrain_path == "default":
            resnet = torchvision.models.resnet50(pretrained=True)
        else:
            resnet = torchvision.models.resnet50(pretrained=False)
            resnet.load_state_dict(torch.load(pretrain_path),strict=False)


        self.resnet_conv = nn.Sequential(resnet.conv1, resnet.bn1, resnet.maxpool)

    def forward(self, rgb):
        rgb_features_map = self.resnet_conv(rgb)
        return rgb_features_map

class IR_Model(nn.Module):
    def __init__(self, pretrain_path=None):
        super(IR_Model, self,).__init__()
        if pretrain_path == "default":
            resnet = torchvision.models.resnet50(pretrained=True)
        else:
            resnet = torchvision.models.resnet50(pretrained = False)
            resnet.load_state_dict(torch.load(pretrain_path),strict=False)

        self.resnet_conv = nn.Sequential(resnet.conv1, resnet.bn1, resnet.maxpool)

    def forward(self, ir):
        ir_features_map = self.resnet_conv(ir)
        return ir_features_map

class Shared_Model(nn.Module):

    def __init__(self, pretrain_path=None):
        super(Shared_Model, self,).__init__()
        if pretrain_path == "default":
            resnet = torchvision.models.resnet50(pretrained=True)
        else:
            resnet = torchvision.models.resnet50(pretrained = False)
            resnet.load_state_dict(torch.load(pretrain_path),strict=False)
        
        resnet.layer4[0].conv2.stride = (1, 1)
        resnet.layer4[0].downsample[0].stride = (1, 1)

        self.resnet_conv = nn.Sequential(resnet.layer1,
                                         resnet.layer2, resnet.layer3, resnet.layer4)

    def forward(self, x):
        features_map = self.resnet_conv(x)
        return features_map

class Classifier(nn.Module):
    def __init__(self, pid_num, dim=2048):
        super(Classifier, self, ).__init__()
        self.pid_num = pid_num
        # self.GAP = GeneralizedMeanPoolingP()
        self.BN = nn.BatchNorm1d(dim)
        self.BN.apply(weights_init_kaiming)

        self.classifier = nn.Linear(dim, self.pid_num, bias=False)
        self.classifier.apply(weights_init_classifier)

        self.l2_norm = Normalize(2)

    def forward(self, features):
        # features = self.GAP(features_map)
        bn_features = self.BN(features.squeeze())
        cls_score = self.classifier(bn_features)
        if self.training:
            return features, cls_score
        else:
            return self.l2_norm(bn_features)


