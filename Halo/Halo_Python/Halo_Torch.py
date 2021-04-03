import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

y_train = tf.one_hot(y_train, depth=10)
y_test = tf.one_hot(y_test, depth=10)

CasterLevel = 32
numTrain = 1000
numTest = 10000

All_Ring = np.zeros(numTrain,CasterLevel)
for i in range(numTrain):
    Img = x_train[i,:,:,1]
    this_Halo = Halo_make(Img,CasterLevel)
    All_Ring[i,:] = this_Halo[i,:]


