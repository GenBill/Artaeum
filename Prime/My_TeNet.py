# -*- coding: UTF-8 -*-
# Neural Network isPrime()
# 导入所需模块
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
from Prime_Gener import prime_gener, prime_32bit

# 生成数据，分别为输入特征和标签
maxTrain = 100
maxTest = 10000
[x_data, y_data] = prime_32bit(maxTest)
# y_data = tf.one_hot(y_data, depth = 2)

x_train = x_data[:maxTrain,:]
y_train = y_data[:maxTrain]
x_test = x_data[maxTrain:,:]
y_test = x_data[maxTrain:]

# Add a channels dimension
# x_train = x_train[..., tf.newaxis]
# x_test = x_test[..., tf.newaxis]

train_ds = tf.data.Dataset.from_tensor_slices(
    (x_train, y_train)).shuffle(10000).batch(25)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(25)

class AnaModel(Model):
  def __init__(self):
    super(AnaModel, self).__init__()
    self.d_00 = Dense(32, activation='relu')
    self.d_01 = Dense(2, activation='softmax')

  def call(self, x):
    x = self.d_00(x)
    return self.d_01(x)

class MyModel(Model):
  def __init__(self):
    super(MyModel, self).__init__()
    self.d_00 = Dense(32, activation='relu')
    self.d_01 = Dense(32, activation='relu')
    self.d_02 = Dense(32, activation='relu')
    self.d_03 = Dense(32, activation='relu')
    self.d_04 = Dense(32, activation='relu')
    self.d_05 = Dense(32, activation='relu')
    self.d_06 = Dense(32, activation='relu')
    self.d_07 = Dense(32, activation='relu')
    self.d_08 = Dense(32, activation='relu')
    self.d_09 = Dense(32, activation='relu')
    self.d_10 = Dense(32, activation='relu')
    self.d_11 = Dense(32, activation='relu')
    self.d_12 = Dense(32, activation='relu')
    self.d_13 = Dense(32, activation='relu')
    self.d_14 = Dense(32, activation='relu')
    self.d_15 = Dense(32, activation='relu')
    self.d_16 = Dense(32, activation='relu')
    self.d_17 = Dense(32, activation='relu')
    self.d_18 = Dense(32, activation='relu')
    self.d_19 = Dense(32, activation='relu')
    self.d_20 = Dense(32, activation='relu')
    self.d_21 = Dense(32, activation='relu')
    self.d_22 = Dense(32, activation='relu')
    self.d_23 = Dense(32, activation='relu')
    self.d_24 = Dense(32, activation='relu')
    self.d_25 = Dense(32, activation='relu')
    self.d_26 = Dense(32, activation='relu')
    self.d_27 = Dense(32, activation='relu')
    self.d_28 = Dense(32, activation='relu')
    self.d_29 = Dense(32, activation='relu')
    self.d_30 = Dense(32, activation='relu')
    self.d_31 = Dense(32, activation='relu')
    self.d_32 = Dense(32, activation='relu')
    self.d_33 = Dense(32, activation='relu')
    self.d_34 = Dense(32, activation='relu')
    self.d_35 = Dense(32, activation='relu')
    self.d_36 = Dense(32, activation='relu')
    self.d_37 = Dense(32, activation='relu')
    self.d_38 = Dense(32, activation='relu')
    self.d_39 = Dense(32, activation='relu')
    self.d_40 = Dense(32, activation='relu')
    self.d_41 = Dense(32, activation='relu')
    self.d_42 = Dense(32, activation='relu')
    self.d_43 = Dense(32, activation='relu')
    self.d_44 = Dense(32, activation='relu')
    self.d_45 = Dense(32, activation='relu')
    self.d_46 = Dense(32, activation='relu')
    self.d_47 = Dense(32, activation='relu')
    self.d_48 = Dense(32, activation='relu')
    self.d_49 = Dense(16, activation='relu')
    self.d_50 = Dense(2, activation='softmax')

  def call(self, x):
    x = self.d_00(x)
    x = self.d_01(x)
    x = self.d_02(x)
    x = self.d_03(x)
    x = self.d_04(x)
    x = self.d_05(x)
    x = self.d_06(x)
    x = self.d_07(x)
    x = self.d_08(x)
    x = self.d_09(x)
    x = self.d_10(x)
    x = self.d_11(x)
    x = self.d_12(x)
    x = self.d_13(x)
    x = self.d_14(x)
    x = self.d_15(x)
    x = self.d_16(x)
    x = self.d_17(x)
    x = self.d_18(x)
    x = self.d_19(x)
    x = self.d_20(x)
    x = self.d_21(x)
    x = self.d_22(x)
    x = self.d_23(x)
    x = self.d_24(x)
    x = self.d_25(x)
    x = self.d_26(x)
    x = self.d_27(x)
    x = self.d_28(x)
    x = self.d_29(x)
    x = self.d_30(x)
    x = self.d_31(x)
    x = self.d_32(x)
    x = self.d_33(x)
    x = self.d_34(x)
    x = self.d_35(x)
    x = self.d_36(x)
    x = self.d_37(x)
    x = self.d_38(x)
    x = self.d_39(x)
    x = self.d_40(x)
    x = self.d_41(x)
    x = self.d_42(x)
    x = self.d_43(x)
    x = self.d_44(x)
    x = self.d_45(x)
    x = self.d_46(x)
    x = self.d_47(x)
    x = self.d_48(x)
    x = self.d_49(x)
    return self.d_00(x)

model = AnaModel()

loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

@tf.function
def train_step(images, labels):
  with tf.GradientTape() as tape:
    predictions = model(images)
    print(labels.shape)
    print(predictions.shape)
    loss = loss_object(labels, predictions)
    # loss  = (predictions - labels)**2
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  train_loss(loss)
  train_accuracy(labels, predictions)

@tf.function
def test_step(images, labels):
  predictions = model(images)
  print(labels.shape)
  print(predictions.shape)
  t_loss = loss_object(labels, predictions)
  # t_loss  = (predictions - labels)**2

  test_loss(t_loss)
  test_accuracy(labels, predictions)

EPOCHS = 1000

for epoch in range(EPOCHS):
  # 在下一个epoch开始时，重置评估指标
  train_loss.reset_states()
  train_accuracy.reset_states()
  test_loss.reset_states()
  test_accuracy.reset_states()

  for images, labels in train_ds:
    train_step(images, labels)

  for test_images, test_labels in test_ds:
    test_step(test_images, test_labels)

  template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
  print (template.format(epoch+1,
                         train_loss.result(),
                         train_accuracy.result()*100,
                         test_loss.result(),
                         test_accuracy.result()*100))