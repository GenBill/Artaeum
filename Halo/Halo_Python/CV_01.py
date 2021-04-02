import cv2 as cv
import numpy as np
import tensorflow as tf

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 取出1张数字图进行测试
# print(x_train.shape)
src = x_train[0,:,:]

# 边缘检测
def edge_demo(image):
    # 高斯模糊：降噪
    blurred = cv.GaussianBlur(image, (3, 3), 0)
    # 灰度转换
    gray = blurred
    # gray = cv.cvtColor(blurred, cv.COLOR_BGR2GRAY)
    # X Gradient，深度cv.CV_16SC1 整型
    xgrad = cv.Sobel(gray, cv.CV_16SC1, 1, 0)
    # Y Gradient
    ygrad = cv.Sobel(gray, cv.CV_16SC1, 0, 1)
    # edge
    edge_output = cv.Canny(xgrad, ygrad, 50, 150)
    # 50, 150  低阈值  高阈值
    # edge_output = cv.Canny(gray, 50, 150)
    cv.imshow("Canny Edge", edge_output)
    # bitwise_and：变成彩色
    # dst = cv.bitwise_and(image, image, mask=edge_output)
    # cv.imshow("Color Edge", dst)


print("--------- Python OpenCV Tutorial ---------")
# src = cv.imread("F:/Matlab/Asriel.jpg")
cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
cv.imshow("input image", src)
edge_demo(src)
cv.waitKey(0)

cv.destroyAllWindows()
