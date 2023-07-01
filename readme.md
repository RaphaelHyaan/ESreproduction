# ES的复现

这个项目旨在复现EchoSpeech，一种基于声学传感的佩戴式眼镜的连续无声语音识别技术。

## 计划
### 时间-距离图的生成
1. 生成信号（已完成）
2. 接受echo（已完成）
3. 生成时频图（已完成）
4. 计算距离（已完成）
### 构建卷积神经网络处理数据
1. 生成测试、训练集（已完成）
2. 搭建训练框架（已完成）
3. 调节参数，迭代优化（已完成）
### 硬件部分实现
- 待补充




## 目前阶段取得的成果
### 信号录制
1. 实现了Cross—Correction
        ![使用np中的相关性结算后得到的结果.png](https://raw.githubusercontent.com/RaphaelHyaan/ESreproduction/main/img/0.012r.png)
2. 实现了echo profile
        - 未差分
        - 差分后
### 模型训练
- image_nr: 在epochs=7下，平均正确率为：87.09%
    
    ![tableresnet18_7_nr_Adam_sch_.jpg](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/79c3d855-40f5-4eeb-ba6a-6775d53412ec/tableresnet18_7_nr_Adam_sch_.jpg)
    
- image_n: 在epochs = 7下，平均正确率为:  81.39%
    
    ![tableresnet18_7_n_Adam_sch_.jpg](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/73ee868a-bcc8-4ca2-818b-312ac1448f2f/tableresnet18_7_n_Adam_sch_.jpg)
    
- imaeg_r: 在epochs = 7下，平均正确率为:  86.08%
    
    ![tableresnet18_7_r_Adam_sch_.jpg](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/3df6eb22-7777-4005-871a-b784cbf21c9a/tableresnet18_7_r_Adam_sch_.jpg)
    
- image_s: 在epochs = 7下，平均正确率为:  66.07%
## 测试集
- images: 原尺寸 （ 1428 ， 1028 ）的包含【打开】【发送】【关机】【呼叫】【结束】【计时】【开机】【闹钟】【你好】【音量】十个标签的测试集
- images_s: 在images框架下，将图片尺寸缩小到 （ 119 ， 119 ） 的训练集包含元素不变
- images_w, image_w2：在images_s的基础上，随机打乱标签顺序的测试集
- images_r: 在images_s的基础上，删除了部分元素的测试集，仅保留【打开】【关机】【结束】【开机】【闹钟】【你好】【音量】
- images_n: 在images_r的基础上，添加了【再见】【one】【two】【three】【four】【five】的测试集
- images_nr: 在images_n的基础上，删除了【five】【闹钟】的测试集

## 细节
### 信号录制部分： FMCW
1. 间断录制
    - 录制: record()
    - 分析: tran_gray()
    - 分析单个样本: analyse()
2. 连续录制
    - 录制: c_record()
    - 分析: c_anodata()
    - 加载: c_load()
    - 分割: c_partition()
    - 分析单个样本: c_test()
    - 自动对齐: c_align()
3. 其他函数
    - 单样本录制: pandr
    - 数据加载:
        - 接受信号: get_data() 
        - 发射信号: get_refer_data()
        - 从npy加载发射信号: load_refer_data() 已弃用
    - 数据图片输出:
        - 发射、接受信号时频图: print_wave_f()
        - 绘制echo profile: print_table()
        - 绘制信号: print_list() 已弃用
    - 信号分析:
        - 计算Cross-Correction: c_corre()
        - 生成距离矩阵: distance_matrix()
    - 文件:
        - load()
        - save()
        - mkdir()
### 模型部分:
1. 模型
    - resnet18 适用于压缩前的图片，本项目主要使用的训练模型
    - resnet resnet50 适用于压缩后的图片
    - lenet_2 适用于大图片的模型，已弃用
    - lenet_3 适用于压缩后图片
    - lenet 适用于压缩后的图片，已弃用
2. 测试集制作: dataset
3. 训练和测试: trainandtest
    - 对于训练集，引入了 ( 0.95 , 1.05 ) 的随机误差
    - 使用了Adam优化器，在Adam优化器自己调节学习率的同时，使用了lr_scheduler额外调节
    - 使用了nn.CrossEntropyLoss()计算损失函数
4. 训练后的模型文件应保存在ckpt文件夹中，但因为目前在测试阶段，并未真正启用


## 更多信息详见Notion
