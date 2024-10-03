import torch
from torch import nn

class DepthwiseSeperableBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.relu = nn.ReLU()
        self.depthwise = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_channels)
        self.bn1 = nn.BatchNorm2d(num_features=in_channels)
        self.conv1x1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv1x1(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(num_features=out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class MobileNetv1(nn.Module):
    def __init__(self, in_channels, a=1, p=1):
        super().__init__()
        self.conv1 = ConvBlock(in_channels=in_channels, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.dws1 = DepthwiseSeperableBlock(in_channels=32, out_channels=64, kernel_size=3)
        self.dws2 = DepthwiseSeperableBlock(in_channels=64, out_channels=128, kernel_size=3, stride=2)
        self.dws3 = DepthwiseSeperableBlock(in_channels=128, out_channels=128, kernel_size=3)
        self.dws4 = DepthwiseSeperableBlock(in_channels=128, out_channels=256, kernel_size=3, stride=2)
        self.dws5 = DepthwiseSeperableBlock(in_channels=256, out_channels=256, kernel_size=3)
        self.dws6 = DepthwiseSeperableBlock(in_channels=256, out_channels=512, kernel_size=3, stride=2)
        self.middleLayer = nn.Sequential(
            DepthwiseSeperableBlock(in_channels=512, out_channels=512, kernel_size=3),
            DepthwiseSeperableBlock(in_channels=512, out_channels=512, kernel_size=3),
            DepthwiseSeperableBlock(in_channels=512, out_channels=512, kernel_size=3),
            DepthwiseSeperableBlock(in_channels=512, out_channels=512, kernel_size=3),
            DepthwiseSeperableBlock(in_channels=512, out_channels=512, kernel_size=3),
        )
        self.dws7 = DepthwiseSeperableBlock(in_channels=512, out_channels=1024, kernel_size=3, stride=2)
        self.dws8 = DepthwiseSeperableBlock(in_channels=1024, out_channels=1024, kernel_size=3) # 논문에는 stride 2로 써있는데 input size와 output size를 보면 stride 1 인 것 같아서 1으로 함.
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(in_features=1024, out_features=1000)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self,x):
        x = self.conv1(x)
        x = self.dws1(x)
        x = self.dws2(x)
        x = self.dws3(x)
        x = self.dws4(x)
        x = self.dws5(x)
        x = self.dws6(x)
        x = self.middleLayer(x)
        x = self.dws7(x)
        x = self.dws8(x)
        x = self.avg_pool(x).squeeze()
        x = self.fc(x)
        x = self.softmax(x)
        return x
        