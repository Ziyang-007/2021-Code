import numpy as np
import os
import gzip
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import models
import tensorflow as tf

# 网络加载数据集
# fashion_mnist = keras.datasets.fashion_mnist
# (train_images, train_labels),(test_images, test_labels) = fashion_mnist.load_data()

# 定义加载数据的函数，data_folder为保存gz数据的文件夹，该文件夹下有4个文件
# 'train-labels-idx1-ubyte.gz', 'train-images-idx3-ubyte.gz',
# 't10k-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz'

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

# 查看训练集的个数
# print(train_images.shape)

# 展示训练集内的某个图像
# plt.imshow(train_images[0])
# plt.show()

# 构建简单的神经网络 (784+1)*128=100480 (128+1)*10=1290
# model = keras.Sequential([
#     keras.layers.Flatten(input_shape=(28, 28)),
#     keras.layers.Dense(128, activation=tf.nn.relu),
#     keras.layers.Dense(10, activation=tf.nn.softmax)
# ])

# 构建卷积神经网络
model = keras.Sequential([
    keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    keras.layers.MaxPool2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPool2D((2, 2)),
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

# 展示神经网络
# model.summary()

# 训练模型 epoch设置训练次数
train_images = train_images/255
model.compile(optimizer=tf.optimizers.Adam(),
              loss=tf.losses.sparse_categorical_crossentropy,
              metrics=['accuracy'])
model.fit(train_images.reshape(-1, 28, 28, 1), train_labels, epochs=10)

# 展示神经网络工作过程的图像,卷积和池化可视化
# layer_outputs = [layer.output for layer in model.layers]
# activation_model = tf.keras.models.Model(inputs=model.input, outputs=layer_outputs)
# pred = activation_model.predict(test_images[0].reshape(-1, 28, 28, 1))
# plt.imshow(pred[0][0, :, :, 2])
# plt.show()

# 测试模型
test_images_scaled = test_images/255
model.evaluate(test_images_scaled.reshape(-1, 28, 28, 1), test_labels)

# 预测测试集的标签
# print(np.argmax(model.predict((test_images[0]/255).reshape(-1, 28, 28, 1))))

