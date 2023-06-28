import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from dataset import MNISTDataset
#from models.resnet import resnet50
from models.lenet_2 import LeNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("将使用%s训练" %(device))
#device = torch.device('cpu') 

# 数据集
train_dataset = MNISTDataset('echospreech/images', train=True, transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)

# 模型
#model = resnet50(num_classes=10).to(device)
model = LeNet().to(device)

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# 训练
epochs = 200
for epoch in range(epochs):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if (i + 1) % 1 == 0:
            print('[epoch: %d, iter: %d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0
torch.save(model.state_dict(), 'echospreech/ckpt/mnist_lenet_02.pth')
#torch.save(model.state_dict(), 'mnist_cnn/ckpt/mnist_resnet.pth')

print('Finished Training')

