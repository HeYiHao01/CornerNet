import math
import numpy as np
import torch
import torch.nn as nn


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, _make_divisible(channel // reduction, 8)),
                nn.ReLU(inplace=True),
                nn.Linear(_make_divisible(channel // reduction, 8), channel),
                h_sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        h_swish()
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        h_swish()
    )


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avgout, maxout], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class InvertedResidual(nn.Module):
    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se, use_hs):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        self.identity = stride == 1 and inp == oup

        if inp == hidden_dim:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # Squeeze-and-Excite
                SELayer(hidden_dim) if use_se else nn.Identity(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(

                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                h_swish() if use_hs else nn.ReLU(inplace=True),

                # dw
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),

                # Squeeze-and-Excite
                SELayer(hidden_dim) if use_se else nn.Identity(),

                h_swish() if use_hs else nn.ReLU(inplace=True),

                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV3(nn.Module):
    def __init__(self, num_classes=1, width_mult=1.):
        super(MobileNetV3, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = [
            #`   k,   t,   c, SE,HS,s
                # 208,208,16 -> 208,208,16
                [3,   1,  16, 0, 0, 1],

                # 208,208,16 -> 104,104,24
                [3,   4,  24, 0, 0, 2],
                [3,   3,  24, 0, 0, 1],

                # 104,104,24 -> 52,52,40
                [5,   3,  40, 1, 0, 2],
                [5,   3,  40, 1, 0, 1],
                [5,   3,  40, 1, 0, 1],

                # 52,52,40 -> 26,26,80
                [3,   6,  80, 0, 1, 2],
                [3, 2.5,  80, 0, 1, 1],
                [3, 2.3,  80, 0, 1, 1],
                [3, 2.3,  80, 0, 1, 1],

                # 26,26,80 -> 26,26,112
                [3,   6, 112, 1, 1, 1],
                [3,   6, 112, 1, 1, 1],

                # 26,26,112 -> 13,13,160
                [5,   6, 160, 1, 1, 2],
                [5,   6, 160, 1, 1, 1],
                [5,   6, 160, 1, 1, 1]
        ]

        input_channel = _make_divisible(16 * width_mult, 8)
        # 416,416,3 -> 208,208,16
        layers = [conv_3x3_bn(3, input_channel, 2)]

        block = InvertedResidual
        for k, t, c, use_se, use_hs, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 8)
            exp_size = _make_divisible(input_channel * t, 8)
            layers.append(block(input_channel, exp_size, output_channel, k, s, use_se, use_hs))
            input_channel = output_channel
        self.features = nn.Sequential(*layers)

        self.conv = conv_1x1_bn(input_channel, exp_size)
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # output_channel = _make_divisible(1280 * width_mult, 8) if width_mult > 1.0 else 1280
        # self.classifier = nn.Sequential(
        #     nn.Linear(exp_size, output_channel),
        #     h_swish(),
        #     nn.Dropout(0.2),
        #     nn.Linear(output_channel, num_classes),
        # )
        # self.SA = SpatialAttention()
        # self.gaussian_filter = self.get_gaussian_kernel()

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)
        # x = self.gaussian_filter * x
        # x = self.SA(x) * x
        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def get_gaussian_kernel(self, kernel_size=20, sigma=2, channels=960, batch_size=16):
        # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
        x_coord = torch.arange(kernel_size)
        x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

        mean = (kernel_size - 1) / 2.
        variance = sigma ** 2.

        # Calculate the 2-dimensional gaussian kernel which is
        # the product of two gaussian distributions for two different
        # variables (in this case called x and y)
        gaussian_kernel = (1. / (2. * math.pi * variance)) * \
                          torch.exp(
                              -torch.sum((xy_grid - mean) ** 2., dim=-1) / (2 * variance)
                          )

        # Make sure sum of values in gaussian kernel equals 1.
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

        # Reshape to 2d depthwise convolutional weight
        gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
        gaussian_kernel = gaussian_kernel.repeat(batch_size, channels, 1, 1)

        # gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, groups=channels,
        #                             bias=False, padding=kernel_size // 2)
        #
        # gaussian_filter.weight.data = gaussian_kernel
        # gaussian_filter.weight.requires_grad = False

        # return gaussian_filter
        return gaussian_kernel


class MobileNetV3_v1(nn.Module):
    def __init__(self, num_classes=1, width_mult=1.):
        super(MobileNetV3_v1, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = [
            #`   k,   t,   c, SE,HS,s
                # 208,208,16 -> 208,208,16
                [3,   1,  16, 0, 0, 1],

                # 208,208,16 -> 104,104,24
                [3,   4,  24, 0, 0, 2],
                [3,   3,  24, 0, 0, 1],

                # 104,104,24 -> 52,52,40
                [5,   3,  40, 1, 0, 2],
                [5,   3,  40, 1, 0, 1],
                [5,   3,  40, 1, 0, 1],

                # 52,52,40 -> 26,26,80
                [3,   6,  80, 0, 1, 2],
                [3, 2.5,  80, 0, 1, 1],
                [3, 2.3,  80, 0, 1, 1],
                [3, 2.3,  80, 0, 1, 1],

                # 26,26,80 -> 26,26,112
                [3,   6, 112, 1, 1, 1],
                [3,   6, 112, 1, 1, 1],

                # 26,26,112 -> 13,13,160
                [5,   6, 160, 1, 1, 2],
                [5,   6, 160, 1, 1, 1],
                [5,   6, 160, 1, 1, 1]
        ]

        input_channel = _make_divisible(16 * width_mult, 8)
        # 416,416,3 -> 208,208,16
        layers = [conv_3x3_bn(3, input_channel, 2)]

        block = InvertedResidual
        for k, t, c, use_se, use_hs, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 8)
            exp_size = _make_divisible(input_channel * t, 8)
            layers.append(block(input_channel, exp_size, output_channel, k, s, use_se, use_hs))
            input_channel = output_channel
        self.features = nn.Sequential(*layers)

        self.conv = conv_1x1_bn(input_channel, exp_size)
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # output_channel = _make_divisible(1280 * width_mult, 8) if width_mult > 1.0 else 1280
        # self.classifier = nn.Sequential(
        #     nn.Linear(exp_size, output_channel),
        #     h_swish(),
        #     nn.Dropout(0.2),
        #     nn.Linear(output_channel, num_classes),
        # )
        self.SA = SpatialAttention()

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)
        x = self.SA(x) * x
        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def mobilenet_v3(pretrained=False, **kwargs):
    # model = MobileNetV3(**kwargs)   # w/o SA
    model = MobileNetV3_v1(**kwargs)
    if pretrained:
        state_dict = torch.load('config/mobilenetv3-large-1cd25616.pth')
        pre_state_dict = {}
        for k, v in state_dict.items():
            if 'classifier' in k:
                continue
            else:
                pre_state_dict[k] = v
        # model.load_state_dict(state_dict, strict=True)
        model.load_state_dict(pre_state_dict, strict=False)
    return model


if __name__ == '__main__':
    model = mobilenet_v3()

    input = torch.randn(16, 3, 640, 640)
    out = model(input)
    print(out.shape)