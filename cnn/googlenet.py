import torch
from torch import nn

class Inception(nn.Module):
    def __init__(self, in_channels, out_channels:list):
        super().__init__()
        self.conv1x1 = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels[0], kernel_size=1),
                nn.ReLU()
            )
        self.conv3x3 = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels[1], kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=out_channels[1], out_channels=out_channels[2], kernel_size=3, padding=1),
                nn.ReLU()
            )
        self.conv5x5 = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels[3], kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=out_channels[3], out_channels=out_channels[4], kernel_size=5, padding=2),
                nn.ReLU()
            )
        self.max_pool = nn.Sequential(
                nn.MaxPool2d(kernel_size=3, padding=1, stride=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels[5], kernel_size=1),
                nn.ReLU()
            )
        
    def forward(self, x):
        x = torch.concat([self.conv1x1(x),self.conv3x3(x),self.conv5x5(x),self.max_pool(x)], dim=1)
        return x

class AuxiliaryClassifier(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.avg_pool = nn.AvgPool2d(kernel_size=5,stride=3)
        self.conv1x1 = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=128, kernel_size=1),
                nn.ReLU()
            )
        self.fc1 = nn.Sequential(
                nn.Linear(in_features=2048,out_features=1024),
                nn.ReLU()
            )
        self.dropout = nn.Dropout(p=0.7)
        self.fc2 = nn.Sequential(
                nn.Linear(in_features=1024,out_features=1000),
                nn.ReLU()
            )
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        x = self.avg_pool(x)
        x = self.conv1x1(x)
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x
    

class InitModule(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.conv7x7 = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, padding=3),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                nn.LocalResponseNorm(size=4)
            )
        self.conv3x3 = nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=192, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.LocalResponseNorm(size=4)
            )
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    def forward(self, x):
        x = self.conv7x7(x)
        x = self.conv3x3(x)
        x = self.max_pool(x)
        return x
    
class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.avg_pool = nn.AvgPool2d(kernel_size=7)
        self.dropout = nn.Dropout(p=0.4)
        self.fc = nn.Linear(in_features=1024, out_features=1000)
        self.softmax = nn.Softmax()
        
    def forward(self,x):
        x = self.avg_pool(x)
        x = self.dropout(x)
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        x = self.fc(x)
        x = self.softmax(x)
        return x

class GoogLeNet(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.init_module = InitModule(in_channels=in_channels)
        self.inception3 = nn.Sequential(
            Inception(in_channels=192, out_channels=[64,96,128,16,32,32]),
            Inception(in_channels=256, out_channels=[128,128,192,32,96,64]),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        )
        self.inception4a = Inception(in_channels=480, out_channels=[192,96,208,16,48,64])
        self.inception4b = Inception(in_channels=512, out_channels=[160,112,224,24,64,64])
        self.inception4c = Inception(in_channels=512, out_channels=[128,128,256,24,64,64])
        self.inception4d = Inception(in_channels=512, out_channels=[112,144,288,32,64,64])
        self.inception4e = Inception(in_channels=528, out_channels=[256,160,320,32,128,128])
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.auxiliary_classifier1 = AuxiliaryClassifier(in_channels=512)
        self.auxiliary_classifier2 = AuxiliaryClassifier(in_channels=528)
        
        self.inception5 = nn.Sequential(
            Inception(in_channels=832, out_channels=[256,160,320,32,128,128]),
            Inception(in_channels=832, out_channels=[384,192,384,48,128,128])
        )
        self.classifer = Classifier()
        
    def forward(self, x):
        x = self.init_module(x)
        x = self.inception3(x)

        x = self.inception4a(x)
        softmax0 = self.auxiliary_classifier1(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        softmax1 = self.auxiliary_classifier2(x)
        x = self.inception4e(x)
        x = self.max_pool(x)
        
        x = self.inception5(x)
        softmax2 = self.classifer(x)
        return softmax0, softmax1, softmax2
        