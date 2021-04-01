# 使用Tensorflow建立神经网络分类horse-or-human

## 数据集horse-or-human

> - 数据集包含了名为horse和human两个文件夹，包含了两种图片，分别是马和人的图片，每种图片大约500张，每个图片大小是300*300，每个像素的取值为3byte,即真彩色图片2的24次方的取值范围

![数据集预览图](https://pic2.zhimg.com/v2-ab4f508a6422e2bdb091e20621dcdb43_1440w.jpg?source=172ae18b)



---

## 加载数据集

1. 创建两个数据生成器，scaling范围为0～1

   ```
   train_datagen = ImageDataGenerator(rescale=1 / 255)
   validation_datagen = ImageDataGenerator(rescale=1 / 255)
   ```

2. 指向训练数据

   ```
   train_generator = train_datagen.flow_from_directory(
       './datasets/HorseOrHuman/horse-or-human',
       target_size=(300, 300),                            # 输出尺寸
       batch_size=32,                                     # 一批训练内数据的个数
       class_mode='binary')                               # 指定二分类
   ```

3. 指向测试数据

   ```
   validation_generator = validation_datagen.flow_from_directory(
       './datasets/HorseOrHuman/validation-horse-or-human',
       target_size=(300, 300),
       batch_size=32,
       class_mode='binary')
   ```

> - 为数据的训练集和测试集添加标签，得到数据的标签为两类，对应horse和human，

```
Found 1027 images belonging to 2 classes.
Found 256 images belonging to 2 classes.
```



---

## 建立卷积神经网络

1. 使用kerastuner优化神经网络结构的参数

> - [Keras Tuner](https://link.zhihu.com/?target=https%3A//github.com/keras-team/keras-tuner) 是一个易于使用、可分布式的超参数优化框架，用于解决执行超参数搜索的痛点。Keras Tuner可以很容易地定义搜索空间，并利用所包含的算法来查找最佳超参数值。 Keras Tuner内置了贝叶斯优化（Bayesian Optimization）、Hyperband和随机搜索（Random Search）算法，同时也便于研究人员扩展，进行新搜索算法的试验。

```
hp = HyperParameters()
```

> - 使用hp.Choice自动选择优化参数，范围为16～64，增加的步长为16，优化卷积层的卷积核数量
> - 添加for循环，确定后续卷积层的层数
> - 使用hp.Int自动选择优化全连接层的输入参数，范围为128～512，增加步长为32
> - 二分类问题，优化策略选择sigmoid，loss函数选择binary_crossentropy，学习率设为0.001

```
def build_model(hp):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(hp.Choice('num_filters_layer0', values=[16, 64], default=16), (3, 3),
                                     activation='relu', input_shape=(300, 300, 3)))
    model.add(tf.keras.layers.MaxPool2D(2, 2))
    for i in range(hp.Int("num_conv_layers", 1, 3)):
        model.add(tf.keras.layers.Conv2D(hp.Choice(f'num_filters_layer{i}', values=[16, 64], default=16), (3, 3),
                                         activation='relu'))
        model.add(tf.keras.layers.MaxPool2D(2, 2))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(hp.Int("hidden_units", 128, 512, step=32), activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=0.001), metrics=['acc'])
    return model
```

> - 生成tuner对象，build_model生成模型，目标使用测试集的acc，估计epochs15次可以达到汇聚
> - 设置参数保存的本地地址horse_human_params

```
tuner = Hyperband(
    build_model,
    objective='val_acc',
    max_epochs=15,
    directory='horse_human_params',
    hyperparameters=hp,
    project_name='my_horse_human_projects'
)
```

> - 开始优化参数

```
tuner.search(train_generator, epochs=10, validation_data=validation_generator)
```

> - 输出优化后最佳的神经网络结构

```
best_hps = tuner.get_best_hyperparameters(1)[0]
print(best_hps.values)
model = tuner.hypermodel.build(best_hps)
model.summary()
```

> - 优化后的神经网络结构

```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 298, 298, 64)      1792      
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 149, 149, 64)      0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 147, 147, 64)      36928     
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 73, 73, 64)        0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 71, 71, 16)        9232      
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 35, 35, 16)        0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 33, 33, 16)        2320      
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 16, 16, 16)        0         
_________________________________________________________________
flatten (Flatten)            (None, 4096)              0         
_________________________________________________________________
dense (Dense)                (None, 448)               1835456   
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 449       
=================================================================
Total params: 1,886,177
Trainable params: 1,886,177
Non-trainable params: 0
_________________________________________________________________
```

> - 神经网络由六层组成：四层的卷积层以及两层的全连接层



---

## 训练神经网络

```
model.fit(train_generator, epochs=10, validation_data=validation_generator)
```

> - 训练结果如表格所示

| Epoch |    loss    |  acc   | val_loss | val_acc |
| :---: | :--------: | :----: | :------: | :-----: |
|   1   |   0.8252   | 0.7118 |  1.6902  | 0.5039  |
|   2   |   0.2658   | 0.8987 |  1.0043  | 0.7734  |
|   3   |   0.1144   | 0.9611 |  2.2868  | 0.7695  |
|   4   |   0.0073   | 0.9990 |  3.5441  | 0.7344  |
|   5   |   0.9119   | 0.9834 |  1.3488  | 0.7227  |
|   6   |   0.0077   | 0.9990 |  2.8768  | 0.7227  |
|   7   |   0.1175   | 0.9718 |  1.2007  | 0.7695  |
|   8   |   0.0034   | 1.0000 |  2.5961  | 0.7578  |
|   9   | 1.0324e-04 | 1.0000 |  2.7516  | 0.7852  |
|  10   |   0.1455   | 0.9757 |  1.1792  | 0.8047  |

> - 发现数据的损失函数波段变化，推测学习率过大，导致训练时错过拟合的位置，将学习率调整为0.0001后训练结果如表格所示

| Epoch |  loss  |  acc   | val_loss | val_acc |
| :---: | :----: | :----: | :------: | :-----: |
|   1   | 0.5389 | 0.7575 |  0.4977  | 0.7773  |
|   2   | 0.2563 | 0.9328 |  0.3915  | 0.8672  |
|   3   | 0.1401 | 0.9611 |  0.8204  | 0.8086  |
|   4   | 0.0892 | 0.9747 |  1.0518  | 0.8047  |
|   5   | 0.0854 | 0.9708 |  0.8878  | 0.8320  |
|   6   | 0.0560 | 0.9825 |  1.0540  | 0.8438  |
|   7   | 0.0529 | 0.9815 |  1.1169  | 0.8320  |
|   8   | 0.0310 | 0.9912 |  1.3099  | 0.8398  |
|   9   | 0.0405 | 0.9873 |  1.3389  | 0.8398  |
|  10   | 0.0322 | 0.9864 |  1.3067  | 0.8398  |

> - 调整学习率后，loss函数的变化更加稳定，并且在10次训练中逐渐减小，测试集的准确度也不断提高，并且稳定在83%左右

---

## 附录

代码使用的库文件

```
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
from tensorflow.keras.optimizers import RMSprop
from kerastuner.tuners import Hyperband
from kerastuner.engine.hyperparameters import HyperParameters
import tensorflow as tf
```