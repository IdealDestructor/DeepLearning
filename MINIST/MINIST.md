# 手写数字识别任务

手写数字识别解决了邮政系统邮政编码识别的问题。手写数字识别常用的数据集是 mnist，该数据集是入门机器学习、深度学习 经典的数据集。  

MNIST 数据集可在 http://yann.lecun.com/exdb/mnist/ 获取, 它包含了四个部分:  
Training set images: train-images-idx3-ubyte.gz (9.9 MB, 解压后 47 MB, 包含 60,000 个样本)  
Training set labels: train-labels-idx1-ubyte.gz (29 KB, 解压后 60 KB, 包含 60,000 个标签)  
Test set images: t10k-images-idx3-ubyte.gz (1.6 MB, 解压后 7.8 MB, 包含 10,000 个样本)  
Test set labels: t10k-labels-idx1-ubyte.gz (5KB, 解压后 10 KB, 包含 10,000 个标签)  

MNIST 数据集来自美国国家标准与技术研究所, National Institute of Standards and Technology (NIST). 训练集 (training set) 由来自 250 个不同人手写的数字构成, 其中 50% 是高中学生, 50% 来自人口普查局 (the Census Bureau) 的工作人员. 测试集(test set) 也是同样比例的手写数字数据.

<center><img src="https://ai-studio-static-online.cdn.bcebos.com/b7c9379268e74660930fb9fdacaff006310a62d3b4334b6a8f05458e791478a6" width="600" hegiht="40" ></center>  
MNIST数据集的发布，吸引了大量科学家训练模型。1998年，LeCun分别用单层线性分类器、多层感知器（Multilayer Perceptron, MLP）和多层卷积神经网络LeNet进行实验，使得测试集的误差不断下降（从12%下降到0.7%）。在研究过程中，LeCun提出了卷积神经网络（Convolutional Neural Network，CNN），大幅度地提高了手写字符的识别能力，也因此成为了深度学习领域的奠基人之一。  

* 任务输入：一系列手写数字图片，其中每张图片都是28x28的像素矩阵。
* 任务输出：经过了大小归一化和居中处理，输出对应的0~9的数字标签。


## 构建手写数字识别的神经网络模型

使用飞桨完成手写数字识别模型任务的代码结构如图所示，与使用飞桨完成房价预测模型任务的流程一致，下面的章节中我们将详细介绍每个步骤的具体实现方法和优化思路。
<br></br>
<center><img src="https://ai-studio-static-online.cdn.bcebos.com/38b467ff3b6e4705b9b4d34c6b13431073e449640ef847f396923475b11c913b" width="800" hegiht="" ></center>
<br></br>

这里的每个模块都有可以配置的不同选项，类似于小朋友插积木，在这个模式固定的框架上更换各种组件，来适应不同需求。  
下面来看一下各个模块的可以插拔的组件。







深度学习的代码结构是“套路”型的。按照固定的套路，替换部分部件就可以完成对应功能，系统的结构如下图所示：
<center><img src="https://ai-studio-static-online.cdn.bcebos.com/82a84e5e3d054c858c65cf5f8988394950e6945c1a5c4d378ce9ead92c0fb91d"  width="800" ></center>

接下来我们展示手写数字识别的代码


```python
#加载飞桨、Numpy和相关类库
import os
import random
import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, Linear
import numpy as np
from PIL import Image

import gzip
import json


```

### 数据处理
### 手写数字的数据处理
MNIST数据集以json格式存储在本地，其数据存储结构如图所示。
<center><img src="https://ai-studio-static-online.cdn.bcebos.com/7d278024d7ac4d6689fdbe0aa1729181699444730e3941d386a55a1ff8ab4276" width="500" hegiht="" ></center>

**data**包含三个元素的列表：**train_set**、**val_set**、 **test_set**，包括50000条训练样本、10000条验证样本、10000条测试样本。每个样本包含手写数字图片和对应的标签。

* **train_set（训练集）**：用于确定模型参数。
* **val_set（验证集）**：用于调节模型超参数（如多个网络结构、正则化权重的最优选择）。
* **test_set（测试集）**：用于估计应用效果（没有在模型中应用过的数据，更贴近模型在真实场景应用的效果）。

**train_set**包含两个元素的列表：**train_images**、**train_labels**。

* **train_images**：[50000, 784]的二维列表，包含50000张图片。每张图片用一个长度为784的向量表示，内容是28\*28尺寸的像素灰度值（黑白图片）。
* **train_labels**：[50000, ]的列表，表示这些图片对应的分类标签，即0-9之间的一个数字。

在本地`./work/`目录下读取文件名称为`mnist.json.gz`的MNIST数据，并拆分成训练集、验证集和测试集，实现方法如下所示。



```python

# 定义数据集读取器
def load_data(mode):

    # 数据文件
    datafile = './work/mnist.json.gz'
    print('loading mnist dataset from {} ......'.format(datafile))
    data = json.load(gzip.open(datafile))
    train_set, val_set, eval_set = data

    # 数据集相关参数，图片高度IMG_ROWS, 图片宽度IMG_COLS
    IMG_ROWS = 28
    IMG_COLS = 28

    if mode == 'train':
        imgs = train_set[0]
        labels = train_set[1]
    elif mode == 'valid':
        imgs = val_set[0]
        labels = val_set[1]
    elif mode == 'eval':
        imgs = eval_set[0]
        labels = eval_set[1]

    imgs_length = len(imgs)

    # 校验
    assert len(imgs) == len(labels), \
          "length of train_imgs({}) should be the same as train_labels({})".format(
                  len(imgs), len(labels))

    print('数据集校验正常')
    print(len(imgs))
    index_list = list(range(imgs_length))

    # 读入数据时用到的batchsize
    BATCHSIZE = 100 #调试修改，调参 128/64

    # 定义数据生成器
    def data_generator():
        if mode == 'train':
            random.shuffle(index_list)
        imgs_list = []
        labels_list = []
        for i in index_list:
            img = np.reshape(imgs[i], [1, IMG_ROWS, IMG_COLS]).astype('float32')
            label = np.reshape(labels[i], [1]).astype('int64')
            imgs_list.append(img) 
            labels_list.append(label)
            if len(imgs_list) == BATCHSIZE and mode == 'train':
                yield np.array(imgs_list), np.array(labels_list)
                imgs_list = []
                labels_list = []

        # 如果剩余数据的数目小于BATCHSIZE，
        # 则剩余数据一起构成一个大小为len(imgs_list)的mini-batch
        if len(imgs_list) > 0 or mode == 'eval':
            yield np.array(imgs_list), np.array(labels_list)

    return data_generator
```


```python
# 卷积网络结构
class MNIST(fluid.dygraph.Layer):
     def __init__(self):
         super(MNIST, self).__init__()
         
         # 定义一个卷积层，使用relu激活函数
         self.conv1 = Conv2D(num_channels=1, num_filters=20, filter_size=5, stride=1, padding=2, act='relu')
         # 定义一个池化层，池化核为2，步长为2，使用最大池化方式
         self.pool1 = Pool2D(pool_size=2, pool_stride=2, pool_type='max')
         # 定义一个卷积层，使用relu激活函数
         self.conv2 = Conv2D(num_channels=20, num_filters=20, filter_size=5, stride=1, padding=2, act='relu')
         # 定义一个池化层，池化核为2，步长为2，使用最大池化方式
         self.pool2 = Pool2D(pool_size=2, pool_stride=2, pool_type='max')
         # 定义一层全连接输出层，输出维度是10
         self.fc = Linear(input_dim=980, output_dim=10, act='softmax')
    # 定义网络的前向计算过程
     def forward(self, inputs):
         x = self.conv1(inputs)
         x = self.pool1(x)
         x = self.conv2(x)
         x = self.pool2(x)
         x = fluid.layers.reshape(x, [x.shape[0], 980])
         x = self.fc(x)
         return x
```

### 配置


```python
with fluid.dygraph.guard():    
    # 声明定义好的线性回归模型
    model = MNIST()
    # 开启模型训练模式
    model.train()
    #调用加载数据的函数
    train_loader = load_data('train')
    # 定义优化算法，这里使用随机梯度下降-SGD
    # 学习率设置为0.01
    optimizer = fluid.optimizer.SGDOptimizer(learning_rate=0.01, parameter_list=model.parameters())

```

    loading mnist dataset from ./work/mnist.json.gz ......


### 训练


```python
#仅修改计算损失的函数，从均方误差（常用于回归问题）到交叉熵误差（常用于分类问题）
with fluid.dygraph.guard():
    EPOCH_NUM = 5
    for epoch_id in range(EPOCH_NUM):
        for batch_id, data in enumerate(train_loader()):
            #准备数据，变得更加简洁
            image_data, label_data = data
            image = fluid.dygraph.to_variable(image_data)
            label = fluid.dygraph.to_variable(label_data)
            
            #前向计算的过程
            predict = model(image)
            
            #计算损失，使用交叉熵损失函数，取一个批次样本损失的平均值
            loss = fluid.layers.cross_entropy(predict, label)
            avg_loss = fluid.layers.mean(loss)
            
            #每训练了200批次的数据，打印下当前Loss的情况
            if batch_id % 50 == 0:
                print("epoch: {}, batch: {}, loss is: {}".format(epoch_id, batch_id, avg_loss.numpy()))
            
            #后向传播，更新参数的过程
            avg_loss.backward()
            optimizer.minimize(avg_loss)
            model.clear_gradients()

    #保存模型参数
    fluid.save_dygraph(model.state_dict(), 'mnist')
```

## 测试
### 数据处理


```python
# 读取一张本地的样例图片，转变成模型输入的格式
def load_image(img_path):
    # 从img_path中读取图像，并转为灰度图
    im = Image.open(img_path).convert('L')
    im.show()
    im = im.resize((28, 28), Image.ANTIALIAS)
    im = np.array(im).reshape(1, 1, 28, 28).astype(np.float32)
    # 图像归一化
    im = 1.0 - im / 255.
    return im

```

### 预测


```python
# 定义预测过程
with fluid.dygraph.guard():
    model = MNIST()
    params_file_path = 'mnist'
    img_path = './work/example_0.jpg'
    # 加载模型参数
    model_dict, _ = fluid.load_dygraph("mnist")
    model.load_dict(model_dict)
    
    model.eval()
    tensor_img = load_image(img_path)
    #模型反馈10个分类标签的对应概率
    results = model(fluid.dygraph.to_variable(tensor_img))
    #取概率最大的标签作为预测输出
    lab = np.argsort(results.numpy())
    print("本次预测的数字是: ", lab[0][-1])
```


```python
# 请编写 预测统计 正确率的代码 并 输出正确率
with fluid.dygraph.guard(): 
    correct_prediction = 0
    prediction = 0
    test_loader = load_data('eval')
    
    model_dict, _ = fluid.load_dygraph("mnist")
    model.load_dict(model_dict)
    
    model.eval()
    for index, data in enumerate(test_loader()):
        #准备数据，变得更加简洁
        image_data, label_data = data
        image = fluid.dygraph.to_variable(image_data)
        label = fluid.dygraph.to_variable(label_data)
        
        #前向计算的过程
        predict = model(image)
        pre = np.argsort(predict.numpy())

        for i in range(len(pre)):
            if label[i][-1] == pre[i][-1]:
                correct_prediction += 1
            prediction += 1
    accuracy = correct_prediction / prediction
    print("正确率: ",accuracy)

```


```python

```
