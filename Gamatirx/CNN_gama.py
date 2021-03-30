import random
import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

y_train = tf.one_hot(y_train, depth=10)
y_test = tf.one_hot(y_test, depth=10)

# Add a channels dimension
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

train_ds = tf.data.Dataset.from_tensor_slices(
    (x_train, y_train)).shuffle(10000).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

class MyModel(Model):
  def __init__(self):
    super(MyModel, self).__init__()
    self.conv1 = Conv2D(32, 3, activation='relu')
    self.flatten = Flatten()
    self.d1 = Dense(128, activation='relu')
    # self.d2 = Dense(10, activation='softmax')
    self.d2 = Dense(11, activation='softmax')
    # 警告：这里需要重定义softmax

  def call(self, x):
    x = self.conv1(x)
    x = self.flatten(x) 
    x = self.d1(x)
    return self.d2(x)
  
  # def my_active(x):
    # return pow(x, 2)
  # def my_active_grad(x):
    # return 2 * x

model = MyModel()

# loss_Matrix
Ma_a = 10
Ma_b = 5
Ma_c = -10
Ma_d = -5
# loss_object
# loss_E2 = tf.square(tf.sub(labels, predictions[:,0:9]))
# loss_Ac = tf.div(1,tf.add(1,loss_E2))
# loss_Se = predictions[:,10]
# loss = loss_Ac*loss_Se*Ma_a + loss_Ac*(1-loss_Se)*Ma_b + (1-loss_Ac)*loss_Se*Ma_c + (1-loss_Ac)*(1-loss_Se)*Ma_d

optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
test_Nope = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

@tf.function
def train_step(images, labels):
  with tf.GradientTape() as tape:
    predictions = model(images)

    loss_E2 = (labels-predictions[:,0:10])**2
    # loss_Ac = 1/(1+loss_E2)
    loss_Ac = 1-loss_E2
    loss_Se = predictions[:,10]
    # loss = loss_object(labels, predictions)
    # loss = -(loss_Ac*loss_Se*Ma_a + loss_Ac*(1-loss_Se)*Ma_b + (1-loss_Ac)*loss_Se*Ma_c + (1-loss_Ac)*(1-loss_Se)*Ma_d)
    loss_Main = Ma_d + tf.reduce_sum((Ma_b-Ma_d)*loss_Ac,axis=1) + (Ma_c-Ma_d)*loss_Se + (Ma_a-Ma_b-Ma_c+Ma_d)*tf.reduce_sum(loss_Ac,axis=1)*loss_Se
    # loss_Main = tf.matmul(loss_Ac,loss_Se)*Ma_a + tf.matmul(loss_Ac,loss_Seinv)*Ma_b + tf.matmul(loss_Acinv,loss_Se)*Ma_c + tf.matmul(loss_Acinv,loss_Seinv)*Ma_d
    loss = -tf.reduce_mean(loss_Main)

  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  train_loss(loss)
  train_accuracy(tf.argmax(labels, axis=1), predictions[:,0:10])

@tf.function
def test_step(images, labels):
  predictions = model(images)
  loss_E2 = (labels-predictions[:,0:10])**2
  # loss_Ac = 1/(1+loss_E2)
  loss_Ac = 1-loss_E2
  loss_Se = predictions[:,10]
  loss_Main = Ma_d + tf.reduce_sum((Ma_b-Ma_d)*loss_Ac,axis=1) + (Ma_c-Ma_d)*loss_Se + (Ma_a-Ma_b-Ma_c+Ma_d)*tf.reduce_sum(loss_Ac,axis=1)*loss_Se
  # loss_Main = tf.matmul(loss_Ac,loss_Se)*Ma_a + tf.matmul(loss_Ac,loss_Seinv)*Ma_b + tf.matmul(loss_Acinv,loss_Se)*Ma_c + tf.matmul(loss_Acinv,loss_Seinv)*Ma_d
  t_loss = -tf.reduce_mean(loss_Main)
  # t_loss = loss_Ac*loss_Se*Ma_a + loss_Ac*(1-loss_Se)*Ma_b + (1-loss_Ac)*loss_Se*Ma_c + (1-loss_Ac)*(1-loss_Se)*Ma_d
  # t_loss = loss_object(labels, predictions)

  test_loss(t_loss)
  test_accuracy(tf.argmax(labels, axis=1), predictions[:,0:10])
  # test_Nope(tf.argmax(labels, axis=1)!=tf.argmax(predictions[:,0:10], axis=1)&predictions[:,10]>0.5)
  test_Nope(tf.argmax(labels, axis=1), predictions[:,0:10])

EPOCHS = 10

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

  template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}, Test Nope: {}'
  print (template.format(epoch+1,
                         train_loss.result(),
                         train_accuracy.result()*100,
                         test_loss.result(),
                         test_accuracy.result()*100,
                         test_Nope.result()*10000))

# Final Test - 输出每次判断的置信度 & 判断准确率的关系
# 2021.03.24 - 15:49
# 快写！

for i in range(1):
  # 好怪，model只接受batch为单位的测试
  this_temp = model(x_test[i:i+32,:,:,:])
  this_pred = this_temp[:,0:10]
  this_trst = this_temp[:,10]
  print(this_temp[:,0:10])
  print(this_temp[:,10])

print('End Sub')
