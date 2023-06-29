#用于随机交换测试集和训练集中的文件
import os
import shutil
import random
def remplace(name):
    path1 = 'echospreech/images_r/train/'+name
    path2 = 'echospreech/images_r/test/'+name
    for i in os.listdir(path2):
        #将所有test中的文件移动至train
        patho = os.path.join(path2,i)
        pathn = os.path.join(path1,i)
        shutil.move(patho,pathn)
    for i in os.listdir(path1):
        if os.path.isdir(os.path.join(path1,i)):
            if random.random() > 20.0/25.0:
                pathn = os.path.join(path2,i)
                patho = os.path.join(path1,i)
                shutil.move(patho,pathn)

for g in os.listdir('echospreech/images_r/test'):
    remplace(g)