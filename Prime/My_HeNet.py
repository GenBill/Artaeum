# -*- coding: UTF-8 -*-
# Neural Network isPrime()
# 导入所需模块
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

from Prime_Gener import prime_32bit

# 生成数据，分别为输入特征和标签
maxTrain = 100
maxTest = 100000
[x_data, y_data] = prime_32bit(maxTest)
x_data = (x_data+1)/2.0
numPrime = sum(y_data)

# 随机打乱数据（因为原始数据是顺序的，顺序不打乱会影响准确率）
# seed: 随机数种子，是一个整数，当设置之后，每次生成的随机数都一样（为方便教学，以保每位同学结果一致）
# np.random.seed(116)  # 使用相同的seed，保证输入特征和标签一一对应
# np.random.shuffle(x_data)
# np.random.seed(116)
# np.random.shuffle(y_data)
# tf.random.set_seed(116)

# 将数据集分割为训练集和测试集，训练集为前100行，测试集为后9900行
x_train = x_data[:maxTrain]
y_train = y_data[:maxTrain]
x_test = x_data[maxTrain:]
y_test = y_data[maxTrain:]

# print(x_train)
# print(y_train)

# 转换x的数据类型，否则后面矩阵相乘时会因数据类型不一致报错
x_train = tf.cast(x_train, tf.float32)
x_test = tf.cast(x_test, tf.float32)

# from_tensor_slices函数使输入特征和标签值一一对应。（把数据集分批次，每个批次batch组数据）
train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)
test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

# 生成神经网络的参数，4个输入特征故，输入层为4个输入节点；因为3分类，故输出层为3个神经元
# 用tf.Variable()标记参数可训练
# 使用seed使每次生成的随机数相同（方便教学，使大家结果都一致，在现实使用时不写seed）
w1 = tf.Variable(tf.random.truncated_normal([32, 8], stddev=0.1, seed=1))
b1 = tf.Variable(tf.random.truncated_normal([8], stddev=0.1, seed=1))

w2 = tf.Variable(tf.random.truncated_normal([8, 2], stddev=0.1, seed=1))
b2 = tf.Variable(tf.random.truncated_normal([2], stddev=0.1, seed=1))

lr = 0.1    # 学习率为0.1
train_loss_results = []  # 将每轮的loss记录在此列表中，为后续画loss曲线提供数据
test_acc0 = []  # 将每轮的acc记录在此列表中，为后续画acc曲线提供数据
test_acc1 = []  # 将每轮的acc记录在此列表中，为后续画acc曲线提供数据
test_acct = []  # 将每轮的acc记录在此列表中，为后续画acc曲线提供数据
num_epoch = 10  # 循环20轮
loss_all = 0  # 每轮分4个step，loss_all记录四个step生成的4个loss的和

# 训练部分
for epoch in range(num_epoch):  #数据集级别的循环，每个epoch循环一次数据集
    # lr = max(min(0.1,lrr**(epoch/100)),0.001)
    for step, (x_train, y_train) in enumerate(train_db):  #batch级别的循环 ，每个step循环一个batch
        with tf.GradientTape() as tape:  # with结构记录梯度信息
            y_temp = tf.matmul(x_train, w1) + b1  # 神经网络乘加运算
            y_temp = tf.nn.sigmoid(y_temp)
            y = tf.matmul(y_temp, w2) + b2  # 神经网络乘加运算
            y = tf.nn.softmax(y)  # 使输出y符合概率分布（此操作后与独热码同量级，可相减求loss）
            y_ = tf.one_hot(y_train, depth=2)  # 将标签值转换为独热码格式，方便计算loss和accuracy
            loss = tf.reduce_mean(tf.square(y_ - y))  # 采用均方误差损失函数mse = mean(sum(y-out)^2)
            loss_all += loss.numpy()  # 将每个step计算出的loss累加，为后续求loss平均值提供数据，这样计算的loss更准确
        # 计算loss对各个参数的梯度
        grads = tape.gradient(loss, [w1,b1,w2,b2])

        # 实现梯度更新 w1 = w1 - lr * w1_grad    b = b - lr * b_grad
        w1.assign_sub(lr * grads[0])  # 参数w1自更新
        b1.assign_sub(lr * grads[1])  # 参数b1自更新
        w2.assign_sub(lr * grads[2])  # 参数w2自更新
        b2.assign_sub(lr * grads[3])  # 参数b2自更新

    # 每个epoch，打印loss信息
    print("Epoch {}, loss: {}".format(epoch, loss_all/4))
    train_loss_results.append(loss_all / 4)  # 将4个step的loss求平均记录在此变量中
    loss_all = 0  # loss_all归零，为记录下一个epoch的loss做准备

    # 测试部分
    # total_correct为预测对的样本个数, total_number为测试的总样本数，将这两个变量都初始化为0
    total_correct, total_correct1, total_number = 0, 0, 0
    for x_test, y_test in test_db:
        # 使用更新后的参数进行预测
        y_temp = tf.matmul(x_test, w1) + b1
        y_temp = tf.nn.softmax(y_temp)
        y = tf.matmul(y_temp, w2) + b2
        y = tf.nn.softmax(y)
        pred = tf.argmax(y, axis=1)  # 返回y中最大值的索引，即预测的分类
        # 将pred转换为y_test的数据类型
        pred = tf.cast(pred, dtype=y_test.dtype)
        whereC = tf.equal(pred, y_test)
        where1 = tf.equal(1, y_test)
        #print(whereC)
        #print(where1)
        # 若分类正确，则correct=1，否则为0，将bool型的结果转换为int型
        correct = tf.cast(tf.equal(pred, y_test), dtype=tf.int32)
        correct1 = tf.cast(tf.equal(where1,whereC), dtype=tf.int32)
        # 将每个batch的correct数加起来
        correct = tf.reduce_sum(correct)
        correct1 = tf.reduce_sum(correct1)
        # 将所有batch中的correct数加起来
        total_correct += int(correct)
        total_correct1 += int(correct1)
        # total_number为测试的总样本数，也就是x_test的行数，shape[0]返回变量的行数
        total_number += x_test.shape[0]
    # 总的准确率等于total_correct/total_number

    acc_0 = (total_correct-total_correct1) / (total_number-numPrime)
    acc_1 = total_correct1 / numPrime
    acc_t = total_correct / total_number
    test_acct.append(acc_t)
    print("acc_0: {}, acc_1: {}, acc_t: {}".format(acc_0, acc_1, acc_t))
    print("--------------------------")
    if total_correct == total_number:
        break

# 绘制 loss 曲线
plt.title('Loss Function Curve')  # 图片标题
plt.xlabel('Epoch')  # x轴变量名称
plt.ylabel('Loss')  # y轴变量名称
plt.plot(train_loss_results, label="$Loss$")  # 逐点画出trian_loss_results值并连线，连线图标是Loss
plt.legend()  # 画出曲线图标
plt.show()  # 画出图像

# 绘制 Accuracy 曲线
plt.title('Acc Curve')  # 图片标题
plt.xlabel('Epoch')  # x轴变量名称
plt.ylabel('Acc')  # y轴变量名称
plt.plot(test_acc0, label="$Accuracy_0$")  # 逐点画出test_acc值并连线，连线图标是Accuracy
plt.plot(test_acc1, label="$Accuracy_1$")  # 逐点画出test_acc值并连线，连线图标是Accuracy
plt.plot(test_acct, label="$Accuracy_t$")  # 逐点画出test_acc值并连线，连线图标是Accuracy
plt.legend()
plt.show()
