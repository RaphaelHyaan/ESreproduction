import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from dataset import MNISTDataset
from models.resnet import resnet50
from models.lenet_3 import LeNet
import numpy as np
import matplotlib.pyplot as plt


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("将使用%s训练" %(device))
#device = torch.device('cpu') 

train_dataset = MNISTDataset('echospreech/images_s', train=True, transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
test_dataset = MNISTDataset('echospreech/images_s', train=False, transform=transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)

# 模型
#model = resnet50(num_classes=10).to(device)
model = LeNet().to(device)
#model.load_state_dict(torch.load('echospreech/ckpt/es_lenet_1403.pth', map_location=device))

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)

#记录loss和accuracy
ll = []
la = []



def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('crorrect/total: %f' %(correct/total))
    return correct/total


def train(foi):
    # 训练
    epochs = 200
    for epoch in range(epochs):
        if epoch % 10 == 0:
            optimizer = optim.SGD(model.parameters(), lr=0.01-epoch*(0.01-0.0001)/500, momentum=0.9)
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
            if (i + 1) % 13 == 0:
                print('[epoch: %d, iter: %d] loss: %.10f' % (epoch + 1, i + 1, running_loss / 13))
                ll.append(running_loss/13)
                la.append(test())
                running_loss = 0.0
    torch.save(model.state_dict(), 'echospreech/ckpt/es_lenet_140'+str(foi)+'.pth')
    #torch.save(model.state_dict(), 'mnist_cnn/ckpt/mnist_resnet.pth')

    nll = np.array(ll)
    nla = np.array(la)
    plt.figure()
    plt.plot(nll)
    plt.plot(nla)
    plt.savefig('ll_la'+str(foi)+'.jpg',dpi = 300)
    # 数据集



for i in range(5):
    train(i)

print('Finished Training')

