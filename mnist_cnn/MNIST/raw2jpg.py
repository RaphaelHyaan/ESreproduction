import os
from PIL import Image
import numpy as np
import gzip
import tqdm

# 定义MNIST数据集文件的路径
data_dir = 'raw'
train_images_file = os.path.join(data_dir, 'train-images-idx3-ubyte.gz')
train_labels_file = os.path.join(data_dir, 'train-labels-idx1-ubyte.gz')
test_images_file = os.path.join(data_dir, 't10k-images-idx3-ubyte.gz')
test_labels_file = os.path.join(data_dir, 't10k-labels-idx1-ubyte.gz')

# 创建保存JPG格式图像的目录
save_dir = 'images'
os.makedirs(save_dir, exist_ok=True)

# 读取训练数据集的图像数据
with gzip.open(train_images_file, 'rb') as f:
    # 跳过文件头信息
    f.read(16)
    buf = f.read(28 * 28 * 60000)  # 读取全部图像数据
    train_data = np.frombuffer(buf, dtype=np.uint8).reshape(-1, 28, 28)

# 读取训练数据集的标签
with gzip.open(train_labels_file, 'rb') as f:
    # 跳过文件头信息
    f.read(8)
    buf = f.read(60000)  # 读取全部标签数据
    train_labels = np.frombuffer(buf, dtype=np.uint8)

# 转换并保存训练数据为JPG格式，并按数字分到不同的文件夹
for i, (img, label) in tqdm.tqdm(enumerate(zip(train_data, train_labels))):
    digit_dir = os.path.join(save_dir, 'train', str(label))
    os.makedirs(digit_dir, exist_ok=True)
    img_path = os.path.join(digit_dir, f'mnist_train_{i}.jpg')
    img_pil = Image.fromarray(img)
    img_pil.save(img_path)

# 读取测试数据集的图像数据
with gzip.open(test_images_file, 'rb') as f:
    # 跳过文件头信息
    f.read(16)
    buf = f.read(28 * 28 * 10000)  # 读取全部图像数据
    test_data = np.frombuffer(buf, dtype=np.uint8).reshape(-1, 28, 28)

# 读取测试数据集的标签
with gzip.open(test_labels_file, 'rb') as f:
    # 跳过文件头信息
    f.read(8)
    buf = f.read(10000)  # 读取全部标签数据
    test_labels = np.frombuffer(buf, dtype=np.uint8)

# 转换并保存测试数据为JPG格式，并按数字分到不同的文件夹
for i, (img, label) in tqdm.tqdm(enumerate(zip(test_data, test_labels))):
    digit_dir = os.path.join(save_dir, 'test', str(label))
    os.makedirs(digit_dir, exist_ok=True)
    img_path = os.path.join(digit_dir, f'mnist_test_{i}.jpg')
    img_pil = Image.fromarray(img)
    img_pil.save(img_path)
