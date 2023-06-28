import torch
import torch.nn as nn

class LeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=5, stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=4),
            nn.Conv2d(16, 32, kernel_size=5, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)

        )
        
        self.classifier = nn.Sequential(
            nn.Linear(32*6*6, 240),
            #nn.Dropout(),
            nn.ReLU(inplace=True),
            nn.Linear(240, 84),
            #nn.Dropout(),
            nn.ReLU(inplace=True),
            nn.Linear(84, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
if __name__ == "__main__":
    # 创建LeNet模型实例
    model = LeNet()

    # 打印LeNet模型结构
    print(model)