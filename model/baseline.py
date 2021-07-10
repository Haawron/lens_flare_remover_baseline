import torch
import torch.nn as nn
import torchvision.models as models
from model.common import ResidualBlock2, Up



class DaconBaseline(nn.Module):

    def __init__(self, args):
        super().__init__() 
        self.backbone = models.resnet101(pretrained=True)
        self.dropout_rate = args.dropout_rate
        self.build_graph()
             
    def build_graph(self):
        self.layer0 = self.backbone.conv1
        self.layer1 = nn.Sequential(
            self.backbone.bn1,
            self.backbone.relu,
            self.backbone.maxpool,
            self.backbone.layer1,
        )
        self.layer2 = self.backbone.layer2
        self.layer3 = self.backbone.layer3
        self.layer4 = self.backbone.layer4
        
        self.lrelu4 = nn.LeakyReLU(0.1, inplace=False)
        self.maxpool4 = nn.MaxPool2d(2)
        self.res4 = ResidualBlock2(1024, 512, self.dropout_rate)

        self.up4 = Up(512, 1024)
        self.up3 = Up(256, 512)
        self.up2 = Up(128, 256)
        self.up1 = Up(64, 64)
        self.up0 = Up(32, concat=False) 
        
        self.dropout0 = nn.Dropout(self.dropout_rate/2)
        self.out = nn.Conv2d(16, 3, kernel_size=1, padding=0)
        
    def forward(self, x):
        # encoder(backbone)
        x = conv1 = self.layer0(x)  # [64 x 128 x 128]
        x = conv2 = self.layer1(x)  # [256 x 64 x 64]
        x = conv3 = self.layer2(x)  # [512 x 32 x 32]
        x = conv4 = self.layer3(x)  # [1024 x 16 x 16]
        x = conv5 = self.layer4(x)  # seems not being used
        
        # decoder
        x = conv4 = self.lrelu4(conv4)
        x = self.maxpool4(x)  # [1024 x 8 x 8]
        x = self.res4(x)  # [512 x 8 x 8]

        x = self.up4(x, conv4)
        x = self.up3(x, conv3)
        x = self.up2(x, conv2)
        x = self.up1(x, conv1)
        x = self.up0(x) 
        
        x = self.dropout0(x)
        x = self.out(x)
        return x

