import torch.nn as nn

class vgg16_conv_block(nn.Module):
    def __init__(self, input_channels, out_channels, rate=0.3, drop=True):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, out_channels, 3 ,1, 1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(rate)
        self.drop =drop

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        if self.drop:
            x = self.dropout(x)
        return(x)

def vgg16_layer(input_channels, out_channels, num, dropout=[0.3, 0.3]):
    """
    Creates a sequence of convolutional blocks + maxpool, as in VGG16.
    Args:
        input_channels: input channels to the first block
        out_channels: number of output channels for this layer
        num_blocks: how many conv blocks before pooling
        dropout: [drop_rate_first, drop_rate_later]
    """

    layers = []
    layers.append(vgg16_conv_block(input_channels, out_channels, dropout[0]))
    for i in range(1, num-1):
        layers.append(vgg16_conv_block(out_channels, out_channels, dropout[1]))
    if num>1:
        layers.append(vgg16_conv_block(out_channels, out_channels, drop=False))
    layers.append(nn.MaxPool2d(2,2))
    return(layers)


class ModifiedVGG16(nn.Module):
    """
        Modified VGG16 for CIFAR-10 or CIFAR-100.
        Args:
            num_classes (int): number of output classes (10 or 100 typically)
    """
    def __init__(self, num_classes=100):
        super(ModifiedVGG16, self).__init__()
        self.features = nn.Sequential(*vgg16_layer(3,64,2,[0.3,0.3]), *vgg16_layer(64,128,2),
                                      *vgg16_layer(128,256,3), *vgg16_layer(256,512,3),
                                      *vgg16_layer(512,512,3))
        self.classifier = nn.Sequential(nn.Dropout(0.2), 
                                        nn.Flatten(), 
                                        nn.Linear(512, 512, bias=True),
                                        nn.BatchNorm1d(512), 
                                        nn.ReLU(inplace=True), 
                                        nn.Linear(512,num_classes, bias=True))
         
    def forward(self, x):
        x = self.features(x)
        x = self.classifier[0](x)  
        x = self.classifier[1](x) 
        x = self.classifier[2](x)   
        x = self.classifier[3](x)  
        x = self.classifier[4](x)  
        x = self.classifier[5](x)     
        return x   
