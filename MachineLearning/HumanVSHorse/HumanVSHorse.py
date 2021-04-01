from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
from tensorflow.keras.optimizers import RMSprop
from kerastuner.tuners import Hyperband
from kerastuner.engine.hyperparameters import HyperParameters
import tensorflow as tf

# 创建两个数据生成器，scaling范围0～1
train_datagen = ImageDataGenerator(rescale=1 / 255)
validation_datagen = ImageDataGenerator(rescale=1 / 255)

# 指向训练数据
train_generator = train_datagen.flow_from_directory(
    './datasets/HorseOrHuman/horse-or-human',
    target_size=(300, 300),  # 输出尺寸
    batch_size=32,  # 一批训练内数据的个数
    class_mode='binary')  # 指定二分类

# 指向测试数据
validation_generator = validation_datagen.flow_from_directory(
    './datasets/HorseOrHuman/validation-horse-or-human',
    target_size=(300, 300),
    batch_size=32,
    class_mode='binary')

# 建立卷积神经网络
# 使用kerastuner自动优化模型的参数
hp = HyperParameters()


# 将创建模型的代码作为函数
# 使用hp.Choice自动选择优化参数，范围为16～64，增加的步长为16，优化卷积层的卷积核数量
# 添加for循环，确定后续卷积层的层数d
# 使用hp.Int自动选择优化全连接层的输入参数，范围为128～512，增加步长为32
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
    model.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=0.0001), metrics=['acc'])
    return model


# 生成tuner对象，build_model生成模型，目标使用测试集的acc，估计epochs15次可以达到汇聚，
# 设置参数保存的本地地址horse_human_params
tuner = Hyperband(
    build_model,
    objective='val_acc',
    max_epochs=15,
    directory='horse_human_params',
    hyperparameters=hp,
    project_name='my_horse_human_projects'
)
# 开始优化参数
# tuner.search(train_generator, epochs=10, validation_data=validation_generator)

# 输出优化参数后的最佳神经网络结构
best_hps = tuner.get_best_hyperparameters(1)[0]
# print(best_hps.values)
model = tuner.hypermodel.build(best_hps)
# model.summary()


# 根据优化后的参数构建的神经网络
# model = keras.Sequential([
#         keras.layers.Conv2D(64, (3, 3), activation='relu',
#                             input_shape=(300, 300, 3)),
#         keras.layers.MaxPool2D(2, 2),
#         keras.layers.Conv2D(64, (3, 3), activation='relu'),
#         keras.layers.MaxPool2D(2, 2),
#         keras.layers.Conv2D(16, (3, 3), activation='relu'),
#         keras.layers.MaxPool2D(2, 2),
#         keras.layers.Conv2D(16, (3, 3), activation='relu'),
#         keras.layers.MaxPool2D(2, 2),
#         keras.layers.Flatten(),
#         keras.layers.Dense(448, activation='relu'),
#         keras.layers.Dense(1, activation='sigmoid')
#     ])

# 训练模型
# history = model.fit(
#     train_generator,
#     epochs=10,
#     verbose=1,
#     validation_data=validation_generator,
#     validation_steps=8)
model.fit(train_generator, epochs=10, validation_data=validation_generator)

