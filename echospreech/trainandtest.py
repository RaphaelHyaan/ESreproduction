import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from dataset import MNISTDataset
from models.resnet import resnet50
from models.resnet18 import resnet18
from models.lenet_3 import LeNet
import numpy as np
import matplotlib.pyplot as plt

num_classe = 10

# 定义随机乘以随机因子的变换
transform_train = transforms.Compose([
    # 随机乘以随机因子的操作
    transforms.RandomAffine(degrees=0, scale=(0.95, 1.05)),
    #随机上下翻转
    transforms.RandomVerticalFlip(p=0),
    # 将PIL.Image对象转换为Tensor
    transforms.ToTensor()
])

transform_test = transforms.Compose([
    # 将PIL.Image对象转换为Tensor
    transforms.ToTensor()
])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("将使用%s训练" %(device))
#device = torch.device('cpu') 

train_dataset = MNISTDataset('echospreech/images_s', train=True, transform=transform_train)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
test_dataset = MNISTDataset('echospreech/images_s', train=False, transform=transform_test)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)
map = train_dataset.label
names = [map[key] for key in range(num_classe)]

# 模型
#model = resnet50(num_classes=10).to(device)
#model = LeNet().to(device)
#model.load_state_dict(torch.load('echospreech/ckpt/es_lenet_1403.pth', map_location=device))
model = resnet18(num_classes=num_classe).to(device)



#记录loss和accuracy
ll = []
la = []
table = np.zeros((num_classe,num_classe))
c = 0


def test():
    correct = 0
    total = 0
    table = np.zeros((num_classe,num_classe))
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


def train(foi,name,epochs = 10):
    # 训练
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    #optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    lr_lambda = lambda epoch: 0.01-epoch*(0.01-0.0001)/(epochs*1.0)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer,lr_lambda=lr_lambda)
    for epoch in range(epochs):


        #optimizer = optim.Adam(model.parameters(), lr=0.01-epoch*(0.01-0.0001)/epochs, momentum=0.9)

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
            if (i + 1) % 8 == 0:
                print('[epoch: %d, iter: %d] loss: %.10f' % (epoch + 1, i + 1, running_loss / 13))
                ll.append(running_loss/13)
                c,table = test()
                la.append(c)
                running_loss = 0.0
        #print("第%d个epoch的学习率：%f" % (epoch, optimizer.param_groups[0]['lr']))
        scheduler.step()
        
    torch.save(model.state_dict(), 'echospreech/ckpt/es_lenet_150'+str(foi)+'.pth')
    #torch.save(model.state_dict(), 'mnist_cnn/ckpt/mnist_resnet.pth')
    '''
    nll = np.array(ll)
    nla = np.array(la)
    plt.figure()
    plt.plot(nll)
    plt.plot(nla)
    plt.title('accuracy:%.5f'%(c))
    plt.savefig('ll_la'+name+str(foi)+'.jpg',dpi = 300)
    # 数据集
    plt.close()''''''
    plt.figure()
    plt.pcolormesh(names,names,table)
    plt.title('accuracy:%.5f'%(c))
    plt.savefig('table'+name+str(foi)+'.jpg',dpi = 300)
    plt.close'''
    return table,c


for i in range(10):
    name = 'resnet18_12_s_Adam_sch_'
    model = resnet18(num_classes=num_classe).to(device)
    table1,c1 = train(i,name,12)
    table += table1
    c += c1
    plt.figure()
    plt.pcolormesh(names,names,table)
    plt.title('accuracy:%.5f'%(c/10))
    plt.savefig('table'+name+'.jpg',dpi = 300)
    plt.close

print('Finished Training')

