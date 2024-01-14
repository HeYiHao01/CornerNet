import math
import torch
from torch import nn

from nets.resnet50 import resnet50, resnet50_Decoder, resnet50_Head
from nets.mobilenet import mobilenet_v3 as mobilenet


class CornerNet_Resnet50(nn.Module):
    def __init__(self, num_classes=1, pretrained=False):
        super(CornerNet_Resnet50, self).__init__()
        self.pretrained = pretrained
        # 512,512,3 -> 16,16,2048
        self.backbone = resnet50(pretrained=pretrained)
        # 16,16,2048 -> 128,128,64
        self.decoder = resnet50_Decoder(2048)
        # -----------------------------------------------------------------#
        #   对获取到的特征进行上采样，进行分类预测和回归预测
        # -----------------------------------------------------------------#
        self.head = resnet50_Head(channel=64, num_classes=num_classes)

        self._init_weights()

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True

    def _init_weights(self):
        if not self.pretrained:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

        self.head.cls_head[-1].weight.data.fill_(0)
        self.head.cls_head[-1].bias.data.fill_(-2.19)

    def forward(self, x):
        feat = self.backbone(x)
        output = self.head(self.decoder(feat))
        return output


class CornerNet_MobilenetV3(nn.Module):
    def __init__(self, num_classes=1, pretrained=False):
        super(CornerNet_MobilenetV3, self).__init__()
        self.pretrained = pretrained

        self.backbone = mobilenet(pretrained=pretrained)

        self.decoder = resnet50_Decoder(960)

        self.head = resnet50_Head(channel=64, num_classes=num_classes)

        self._init_weights()

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True

    def _init_weights(self):
        if not self.pretrained:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

        self.head.cls_head[-1].weight.data.fill_(0)
        self.head.cls_head[-1].bias.data.fill_(-2.19)

    def forward(self, x):
        feat = self.backbone(x)
        output = self.head(self.decoder(feat))
        return output


if __name__ == '__main__':
    # model = CornerNet_Resnet50(1, pretrained=False)
    model = CornerNet_MobilenetV3(1, pretrained=False)
    input = torch.randn((8, 3, 512, 512))
    output = model(input)
    print(output[0].shape, output[1].shape)