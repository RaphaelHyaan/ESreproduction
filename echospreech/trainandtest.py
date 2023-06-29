import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from dataset import MNISTDataset
from models.resnet import resnet50
from models.lenet_3 import LeNet
import numpy as np
import matplotlib.pyplot as plt


# 定义随机乘以随机因子的变换
transform = transforms.Compose([
    # 随机乘以随机因子的操作
    transforms.RandomAffine(degrees=0, scale=(0.95, 1.05)),
    # 将PIL.Image对象转换为Tensor
    transforms.ToTensor()
])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("将使用%s训练" %(device))
#device = torch.device('cpu') 

train_dataset = MNISTDataset('echospreech/images_w2', train=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
test_dataset = MNISTDataset('echospreech/images_w2', train=False, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)
map = train_dataset.label
names = [map[key] for key in range(10)]

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
table = np.zeros((10,10))
c = 0


def test():
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
            correct += (predicted == labels).sum().item()
            for i,j in zip(labels,predicted):
                table[i][j] += 1
    print('crorrect/total: %f' %(correct/total))
    return correct/total,table


def train(foi,name,epochs = 35):
    # 训练
    
    for epoch in range(epochs):


        optimizer = optim.SGD(model.parameters(), lr=0.01-epoch*(0.01-0.0001)/epochs, momentum=0.9)

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
                c,table = test()
                la.append(c)
                running_loss = 0.0
    torch.save(model.state_dict(), 'echospreech/ckpt/es_lenet_150'+str(foi)+'.pth')
    #torch.save(model.state_dict(), 'mnist_cnn/ckpt/mnist_resnet.pth')

    nll = np.array(ll)
    nla = np.array(la)
    plt.figure()
    plt.plot(nll)
    plt.plot(nla)
    plt.title('accuracy:%.5f'%(c))
    plt.savefig('ll_la'+name+str(foi)+'.jpg',dpi = 300)
    # 数据集
    plt.close()
    plt.figure()
    plt.pcolormesh(names,names,table)
    plt.title('accuracy:%.5f'%(c))
    plt.savefig('table'+name+str(foi)+'.jpg',dpi = 300)
    plt.close


for i in range(5):
    train(i,'LEnet50_35_w_',25)

print('Finished Training')

