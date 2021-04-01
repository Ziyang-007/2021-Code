# 使用神经网络工具TensorFlow训练FashionMNIST
## 数据集FashionMNIST 
>- FashionMNIST 是一个替代 MNIST 手写数字集的图像数据集。 其涵盖了来自 10 种类别的共 7 万个不同商品的正面图片。  
>- FashionMNIST 的大小、格式和训练集/测试集划分与原始的 MNIST 完全一致。60000/10000 的训练测试数据划分，28x28 的灰度图片。

![数据集预览图](https://pic1.zhimg.com/80/v2-e60bed6967b69572478af1a390131604_1440w.jpg)
- - -
## 加载数据集
1. 网络加载数据集
```
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels),(test_images, test_labels) = fashion_mnist.load_data()
```


2. 本地加载数据集

>定义加载数据的函数，./datasets/fashion/为保存gz数据的文件夹，该文件夹下有4个文件
>- 'train-labels-idx1-ubyte.gz',
>- 'train-images-idx3-ubyte.gz',
>- 't10k-labels-idx1-ubyte.gz', 
>- 't10k-images-idx3-ubyte.gz'

```
def load_data(data_folder):
    files = [
        'train-labels-idx1-ubyte.gz', 'train-images-idx3-ubyte.gz',
        't10k-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz'
    ]

    paths = []
    for fname in files:
        paths.append(os.path.join(data_folder, fname))
    
    with gzip.open(paths[0], 'rb') as lbpath:
        y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)
    
    with gzip.open(paths[1], 'rb') as imgpath:
        x_train = np.frombuffer(
            imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28)
    
    with gzip.open(paths[2], 'rb') as lbpath:
        y_test = np.frombuffer(lbpath.read(), np.uint8, offset=8)
    
    with gzip.open(paths[3], 'rb') as imgpath:
        x_test = np.frombuffer(
            imgpath.read(), np.uint8, offset=16).reshape(len(y_test), 28, 28)
    
    return (x_train, y_train), (x_test, y_test)


(train_images, train_labels), (test_images, test_labels) = load_data('./datasets/fashion/')
```

3. 查看训练集的个数
```
print(train_images.shape)
```

4. 展示训练集内的某个图像
```
plt.imshow(train_images[0])
plt.show()
```
![训练集内某个图像预览](https://img-blog.csdnimg.cn/2018121821364581.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzI4ODY5OTI3,size_16,color_FFFFFF,t_70)

- - -
## 构建神经网络
>简单神经网络添加flatten层、全连接层  
卷积神经网络添加卷积层、池化层、flatten层和全连接层  
全连接层最后输出10，分别对应FashionMNIST的10个类别  

>构建完成后可以使用 `model.summary()` 预览神经网络的结构
1. 构建简单的神经网络
```
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])
```

>神经网络结构预览  
>(784+1)*128=100480 (128+1)*10=1290
```
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
flatten (Flatten)            (None, 784)               0         
_________________________________________________________________
dense (Dense)                (None, 128)               100480    
_________________________________________________________________
dense_1 (Dense)              (None, 10)                1290      
=================================================================
Total params: 101,770
Trainable params: 101,770
Non-trainable params: 0
_________________________________________________________________

```

2. 构建卷积神经网络
```
model = keras.Sequential([
    keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    keras.layers.MaxPool2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPool2D((2, 2)),
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])
```
>卷积神经网络结构预览
```
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 26, 26, 64)        640       
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 13, 13, 64)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 11, 11, 64)        36928     
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 5, 5, 64)          0         
_________________________________________________________________
flatten (Flatten)            (None, 1600)              0         
_________________________________________________________________
dense (Dense)                (None, 128)               204928    
_________________________________________________________________
dense_1 (Dense)              (None, 10)                1290      
=================================================================
Total params: 243,786
Trainable params: 243,786
Non-trainable params: 0
_________________________________________________________________
```


---
## 训练神经网络
>训练模型,epoch设置训练次数,设置epoch为10，训练10组训练集  
>多分类问题loss函数使用交叉熵更为合适，优化策略使用Adam
```
train_images = train_images/255
model.compile(optimizer=tf.optimizers.Adam(),
              loss=tf.losses.sparse_categorical_crossentropy,
              metrics=['accuracy'])
model.fit(train_images.reshape(-1, 28, 28, 1), train_labels, epochs=10)
```
---
## 测试模型
1. 使用训练集测试模型的准确度
```
test_images_scaled = test_images/255
model.evaluate(test_images_scaled.reshape(-1, 28, 28, 1), test_labels)
```
>测试结果
```
#简单的神经网络
313/313 [==============================] - 0s 500us/step - loss: 0.3448 - accuracy: 0.8847
#卷积神经网络
313/313 [==============================] - 1s 3ms/step - loss: 0.2859 - accuracy: 0.9111
```

2. 使用模型预测测试集的标签
>预测测试集第一个图像的标签
```
print(np.argmax(model.predict((test_images[0]/255).reshape(-1, 28, 28, 1))))
```
>测试结果
```
9
```


---
## 模型训练结果对比
>简单的神经网络和卷积神经网络，损失函数loss和准确度对比

|  Epoch | 简单的神经网络(loss/accuracy) | 卷积神经网络(loss/accuracy) |
| :----: |       :----:       |     :----:    |
|   1    | 0.4960/0.8269      | 0.4513/0.8354 |
|   2    | 0.3750/0.8658      | 0.3036/0.8893 |
|   3    | 0.3376/0.8771      | 0.2563/0.9050 |
|   4    | 0.3154/0.8840      | 0.2221/0.9174 |
|   5    | 0.2960/0.8920      | 0.1951/0.9274 |
|   6    | 0.2802/0.8958      | 0.1745/0.9354 |
|   7    | 0.2695/0.9002      | 0.1540/0.9430 |
|   8    | 0.2602/0.9028      | 0.1367/0.9492 |
|   9    | 0.2494/0.9062      | 0.1193/0.9543 |
|   10   | 0.2406/0.9097      | 0.1077/0.9593 |

> 由表格可以得出结论：  
> 1. 两个神经网络的训练结果都不错，loss函数的数值都在不断降低
> 2. 卷积神经网络比简单的神经网络效果更好，证明一定情况下神经网络的宽度和深度更大的神经网络效果更好，但是所需的性能要求更高，训练的时间更长。 


---
## 附录
代码使用的库文件
```
import numpy as np
import os
import gzip
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import models
import tensorflow as tf
```
