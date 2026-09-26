import torch
import torch.nn as nn

# (Keep your existing vgg16_conv_block exactly as is)
class vgg16_conv_block(nn.Module):
    def __init__(self, input_channels, out_channels, rate=0.3, drop=True):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, out_channels, 3, 1, 1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(rate)
        self.drop = drop

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        if self.drop:
            x = self.dropout(x)
        return x

# (Keep your existing vgg16_layer exactly as is)
def vgg16_layer(input_channels, out_channels, num, dropout=[0.3, 0.3]):
    layers = []
    layers.append(vgg16_conv_block(input_channels, out_channels, dropout[0]))
    for i in range(1, num-1):
        layers.append(vgg16_conv_block(out_channels, out_channels, dropout[1]))
    if num > 1:
        layers.append(vgg16_conv_block(out_channels, out_channels, drop=False))
    layers.append(nn.MaxPool2d(2,2))
    return layers


# --- THE NEW UNIFIED KD CLASS ---
class VGG_KD(nn.Module):
    """
    Unified VGG for Knowledge Distillation.
    cfg is a list defining the number of blocks per stage.
    """
    def __init__(self, cfg, num_classes=100):
        super(VGG_KD, self).__init__()
        
        # Break the network into 5 distinct stages to capture intermediate features
        self.stage1 = nn.Sequential(*vgg16_layer(3, 64, cfg[0], [0.3, 0.3]))
        self.stage2 = nn.Sequential(*vgg16_layer(64, 128, cfg[1]))
        self.stage3 = nn.Sequential(*vgg16_layer(128, 256, cfg[2]))
        self.stage4 = nn.Sequential(*vgg16_layer(256, 512, cfg[3]))
        self.stage5 = nn.Sequential(*vgg16_layer(512, 512, cfg[4]))

        self.classifier = nn.Sequential(
            nn.Dropout(0.2), 
            nn.Flatten(), 
            nn.Linear(512, 512, bias=True),
            nn.BatchNorm1d(512), 
            nn.ReLU(inplace=True), 
            nn.Linear(512, num_classes, bias=True)
        )
         
    def forward(self, x):
        # Forward pass capturing stage outputs
        f1 = self.stage1(x)
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)
        f4 = self.stage4(f3)
        f5 = self.stage5(f4)
        
        out = self.classifier(f5)
        
        # Format the output to match our Distiller framework
        feats = {
            "feats": [f1, f2, f3, f4, f5],
            "pooled_feat": f5.view(f5.size(0), -1) # Flattened feature for CRD/RKD if needed
        }
        
        return out, feats

# --- HELPER FUNCTIONS TO INSTANTIATE TEACHER/STUDENT ---

def VGG16_Teacher(num_classes=100):
    """The original deep VGG16 structure"""
    return VGG_KD(cfg=[2, 2, 3, 3, 3], num_classes=num_classes)

def VGG8_Student(num_classes=100):
    """A shallow, high-speed Student VGG structure"""
    return VGG_KD(cfg=[1, 1, 1, 1, 1], num_classes=num_classes)

def VGG19_Teacher(num_classes=100):
    """The deeper VGG19 structure (stages 2-2-4-4-4).

    Bigger same-family teacher for a VGG16 student. Module order and every
    tensor shape match Model.VGG16.ModifiedVGG19, so the order-mapping
    fallback in util.create_st_te_model loads a vanilla VGG19 checkpoint the
    same way it already does for VGG16.
    """
    return VGG_KD(cfg=[2, 2, 4, 4, 4], num_classes=num_classes)
