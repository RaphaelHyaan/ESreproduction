#用于压缩图片
import os
import torchvision.transforms as transforms
from PIL import Image

def images(root_dir):
    resize = transforms.Resize((119,119))
    for label in os.listdir(root_dir):
        label_dir = os.path.join(root_dir, label)
        if os.path.isdir(label_dir):
            for sample_name in os.listdir(label_dir):
                sample_path = os.path.join(label_dir, sample_name)
                if os.path.isdir(sample_path):
                    for image_name in os.listdir(sample_path):
                        image_path = os.path.join(sample_path,image_name)
                        image = Image.open(image_path).convert('L')
                        image = resize(image)
                        image.save(image_path)
