import torch
import torch.nn as nn

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, is_last=False):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
            
        # KD Modification: Flag to capture pre-activations on the last block of a stage
        self.is_last = is_last 
        
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        
        # KD Modification: Capture the feature map BEFORE the final ReLU
        preact = out 
        out = self.relu(out)

        if self.is_last:
            return out, preact
        else:
            return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=100, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
            
        self.groups = groups
        self.base_width = width_per_group
        
        # CIFAR-style initial layer (3x3 conv, no maxpool)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        
        # Network Stages
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
                                       
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        
        if dilate:
            self.dilation *= stride
            stride = 1
            
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        # KD Modification: If there is only 1 block, it is the last block. 
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, is_last=(blocks == 1)))
        self.inplanes = planes * block.expansion
        
        for i in range(1, blocks):
            # KD Modification: Flag the final block in the loop
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, is_last=(i == blocks - 1)))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        f0 = x

        # Unpack the tuples returned by the modified layers
        x, f1_pre = self.layer1(x)
        f1 = x
        x, f2_pre = self.layer2(x)
        f2 = x
        x, f3_pre = self.layer3(x)
        f3 = x
        x, f4_pre = self.layer4(x)
        f4 = x

        x = self.avgpool(x)
        avg = torch.flatten(x, 1)
        out = self.fc(avg)

        # Construct the dictionary for FitNet
        feats = {
            "feats": [f0, f1, f2, f3, f4],
            "preact_feats": [f0, f1_pre, f2_pre, f3_pre, f4_pre],
            "pooled_feat": avg
        }

        # Return identical signature to original KD code
        return out, feats


# ==========================================
# Model Instantiation Functions
# ==========================================

def _resnet(block, layers, **kwargs):
    model = ResNet(block, layers, **kwargs)
    return model

def ResNet18_dist(**kwargs):
    """The Teacher Model: 4 stages, 2 blocks per stage."""
    return _resnet(BasicBlock, [2, 2, 2, 2], **kwargs)

def ResNet10_dist(**kwargs):
    """The Student Model: 4 stages, 1 block per stage."""
    return _resnet(BasicBlock, [1, 1, 1, 1], **kwargs)

def ResNet34_dist(**kwargs):
    """Bigger same-family teacher: 4 stages, [3, 4, 6, 3] BasicBlocks.

    Module names follow Model.ResNet_18.ResNet34 exactly, so a vanilla
    ResNet-34 checkpoint (main_train_teacher.py / main_overlap) loads with
    strict=True, the same way ResNet18_dist loads a vanilla ResNet-18.
    Per-stage channels (64/128/256/512) equal ResNet-18's, so a FitNet hint
    would need no channel projection either.
    """
    return _resnet(BasicBlock, [3, 4, 6, 3], **kwargs)

# ==========================================
# Quick Sanity Check
# ==========================================
if __name__ == "__main__":
    x = torch.randn(2, 3, 32, 32)
    
    teacher = ResNet18_dist(num_classes=100)
    student = ResNet10_dist(num_classes=100)
    
    t_logits, t_feats = teacher(x)
    s_logits, s_feats = student(x)
    
    print("Teacher Logits Shape:", t_logits.shape)
    print("Student Logits Shape:", s_logits.shape)
    print(f"Teacher Stages Captured: {len(t_feats['feats'])}")
    print(f"Student Stages Captured: {len(s_feats['feats'])}")