import math
import torch
import ptens
import torch.nn as nn

def conv3x3(in_p, out_p, stride = 1, groups = 1):
    return nn.Conv2d(in_p, out_p, kernel_size = 3,
                     stride = stride, padding = 1, bias = False)

def conv1x1(in_p, out_p, stride = 1):
    return nn.Conv2d(in_p, out_p, kernel_size = 1,
                     stride = stride, bias = False)

class BB(nn.Module):
    expansion = 1
    def __init__(self, in_p, p, stride = 1, ps = None):
        super(BB, self).__init__()
        self.conv1 = conv3x3(in_p, p, stride)
        self.bn1 = nn.BatchNorm2d(p)
        self.relu = nn.ReLU(inplace = True)
        self.conv2 = conv3x3(p, p)
        self.bn2 = nn.BatchNorm2d(p)
        self.ps = ps
        self.stride = stride

    def forward(self, x):
        res = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.ps is not None:
            res = self.ps(x)
        out += res
        out = self.relu(out)
        return out

class BN(nn.Module):
    expansion = 4
    def __init__(self, in_p, p, stride = 1, ps = None, groups = 1,
                 base_width = 64, norm_layer = None):
        super(BN, self).__init__()
        self.conv1 = nn.Conv2d(in_p, p, kernel_size = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(p)
        self.conv2 = nn.Conv2d(p, p, kernel_size = 3, stride = stride,
                               padding = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(p)
        self.conv3 = nn.Conv2d(p, p*4, kernel_size = 1, bias = False)
        self.relu = nn.ReLU(in_p = True)
        self.ps = ps
        self.stride = stride

    def forward(self, x):
        res = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.ps is not None:
            res = self.ps(x)
        out += res
        out = self.relu(out)
        return out

class RN(nn.Module):
    def __init__(self, block, layers, n_classes = 1000, zero_init_res = False, groups = 1, width_per_group = 64, norm_layer = None):
        super(RN, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.in_p = 64
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.in_p, kernel_size = 7, stride = 2, padding = 3, bias = False)
        self.bn1 = norm_layer(self.in_p)
        self.relu = nn.ReLU(inplace = True)
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        self.layer1 = self.layering(block, 64, layers[0], norm_layer=norm_layer)
        self.layer2 = self.layering(block, 128, layers[1], stride=2, norm_layer=norm_layer)
        self.layer3 = self.layering(block, 256, layers[2], stride=2, norm_layer=norm_layer)
        self.layer4 = self.layering(block, 512, layers[3], stride=2, norm_layer=norm_layer)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, n_classes)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.w, mode = 'fan_out', nonlinearity = 'relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.w, 1))
                nn.init.constant_(m.bias, 0)

        if zero_init_res:
            for m in self.modules():
                if isinstance(m,BN):
                    nn.init.constant_(m.bn3.w, 0)
                elif isinstance(m, BB):
                    nn.init.constant_(m.bn2.w, 0)

        def layering(self, block, p, blocks, stride=1, norm_layer = None):
            if norm_layer is None:
                norm_layer = nn.BatchNorm2d
            ps = None
            if stride != 1 or self.in_p != p * block.expansion:
                ps = nn.Sequential(conv1x1(self.in_p, p*block.expansion, stride),
                                   norm_layer(p, block.expansion),)
            layers = []
            layers.append(block(self.in_p, p, stride, ps, self.groups,
                                self.base_width, norm_layer))
            self.in_p = p * block.expansion
            for _ in range(1, blocks):
                layers.append(block(self.in_p, p, groups = self.groups,
                                    base_width = self.base_width, norm_layer = norm_layer))
            return nn.Sequential(*layers)

        def forward(self, x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.avgpool(x)
            x = x.view(x.size(0),-1)
            return x 
                              
def rn(**kwargs):
    model = RN(BN,[3,4,5,6],**kwargs)
    return model
                 
