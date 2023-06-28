import os
from PIL import Image
from torch.utils.data import Dataset

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
        
        # 获取所有图像文件路径和对应的标签
        for label in os.listdir(root_dir):
            label_dir = os.path.join(root_dir, label)
            if os.path.isdir(label_dir):
                for image_name in os.listdir(label_dir):
                    image_path = os.path.join(label_dir, image_name)
                    self.image_paths.append(image_path)
                    self.labels.append(int(label))
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert('L')  # 灰度图像
        label = self.labels[index]
        
        if self.transform is not None:
            image = self.transform(image)
            
        return image, label

if __name__ == "__main__":
    from torchvision.transforms import ToTensor

    # 创建数据集实例
    dataset = MNISTDataset(root_dir='MNIST/images', transform=ToTensor())
    print(f"Dataset Size: {len(dataset)}")

    # 获取数据样本
    image, label = dataset[0]
    print(f"Image Size: {image.shape}")
    print(f"Label: {label}")
    import cv2
    cv2.imshow('image', image.numpy()[0])
    cv2.waitKey(0)

    # # 使用DataLoader加载数据集
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
