import torch
from torchvision import transforms
from dataset import MNISTDataset
from models.resnet import resnet50
from models.lenet_3 import LeNet
import numpy as np
import matplotlib.pyplot as plt
#import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu') 

# 数据集
test_dataset = MNISTDataset('echospreech/images_s', train=False, transform=transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)

# 模型和权重
model = LeNet().to(device)
#model = resnet50(num_classes=10).to(device)
#model.load_state_dict(torch.load('mnist_cnn/ckpt/mnist_resnet50.pth', map_location=device))
model.load_state_dict(torch.load('echospreech/ckpt/es_lenet_1304.pth', map_location=device))
# 测试
correct = 0
total = 0
table = np.zeros((10,10))
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        for i,j in zip(labels,predicted):
            table[i][j] += 1

        correct += (predicted == labels).sum().item()
        print('crorrect:%d; total:%d' %(correct, total))

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
