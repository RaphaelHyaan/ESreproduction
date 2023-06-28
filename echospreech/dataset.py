import os
from PIL import Image
from torch.utils.data import Dataset
import torch
import torchvision.transforms as transforms

class MNISTDataset(Dataset):
    def __init__(self, root_dir, transform=None, train=True):
        if train:
            root_dir = os.path.join(root_dir, 'train')
        else:
            root_dir = os.path.join(root_dir, 'test')

        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.label = {}
        
        # 获取所有图像文件路径和对应的标签
        i = 0
        for label in os.listdir(root_dir):
            
            label_dir = os.path.join(root_dir, label)
            if os.path.isdir(label_dir):
                for sample_name in os.listdir(label_dir):
                    sample_path = os.path.join(label_dir, sample_name)
                    if os.path.isdir(sample_path):
                        sample_image = []#一个样本中的四个图像
                        for image_name in os.listdir(sample_path):
                            image_path = os.path.join(sample_path,image_name)
                            sample_image.append(image_path)
                        self.image_paths.append(sample_image)
                        self.labels.append(i)
            self.label[label] = i
            i+=1 
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        image_path = self.image_paths[index]
        images = []
        for i in range(4):
            image = Image.open(image_path[i]).convert('L')
            
            images.append(image)
        
        if self.transform is not None:
            for i in range(4):
                images[i] = self.transform(images[i])#在训练之前压缩图片
        images = torch.cat(images,dim = 0)
        #这里暂时先默认使用totenser，如果将来要换为其他转化器，需要自己编写转换器而将这部分代码还原回最开始的形态
        
        label = self.labels[index]
        

        return images, label

if __name__ == "__main__":
    from torchvision.transforms import ToTensor
    from torchvision.transforms import ToTensor

    # 创建数据集实例
    dataset = MNISTDataset(root_dir='echospreech/images', transform=ToTensor())
    print(f"Dataset Size: {len(dataset)}")

    # 获取数据样本
    image, label = dataset[0]
    print(f"Image Size: {image[0].shape}")
    print(f"Label: {label}")
    import cv2
    cv2.imshow('image', image.numpy()[0])
    cv2.waitKey(0)