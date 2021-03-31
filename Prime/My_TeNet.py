# -*- coding: UTF-8 -*-
# Neural Network isPrime()
# 导入所需模块
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D

from Prime_Gener import prime_gener, prime_32bit

# 生成数据，分别为输入特征和标签
maxTrain = 100
maxTest = 10000
[x_data, y_data] = prime_32bit(maxTest)

