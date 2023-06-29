import os
import random

# 生成不重复的十以内的随机数组
array = random.sample(range(0, 10), 10)

# 获取文件夹路径
folder_path1 = 'echospreech/images_w2/test'
folder_path2 = 'echospreech/images_w2/train'

# 遍历文件夹中的子文件夹
for folder_name1,folder_name2 in zip(os.listdir(folder_path1),os.listdir(folder_path2)):
    folder1 = os.path.join(folder_path1, folder_name1)
    folder2 = os.path.join(folder_path2, folder_name2)
    if os.path.isdir(folder1):
        # 更新文件夹中的子文件夹名字
        num = str(array.pop())
        
        new_name1 = num + '_' + folder_name1
        new_name2 = num + '_' + folder_name2
        '''
        new_name1 = folder_name1[2:]
        new_name2 = folder_name2[2:]'''

        new_folder1 = os.path.join(folder_path1, new_name1)
        new_folder2 = os.path.join(folder_path2, new_name2)
        os.rename(folder1, new_folder1)
        os.rename(folder2, new_folder2)