#用于测试集的建立
import os
import shutil
def make_dir(path):
    for image in os.listdir(path):
        name = image.split('_')
        if name[0] == image:
            name = image.split('.')
            new_path = os.path.join(path,name[0])
            os.mkdir(new_path)

        else:
            new_path = os.path.join(path,name[0])
            new_path = os.path.join(new_path,image)
            old_path = os.path.join(path,image)
            shutil.move(old_path,new_path)


for path in os.listdir('echospreech/images/train'):
    if path != 'dakai':
        make_dir('echospreech/images/train/'+path)
