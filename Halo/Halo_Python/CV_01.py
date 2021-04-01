import cv2 as cv
import numpy as np

#边缘检测
def edge_demo(image):
    #高斯模糊：降噪
    blurred = cv.GaussianBlur(image, (3, 3), 0)
    #灰度转换
    gray = cv.cvtColor(blurred, cv.COLOR_BGR2GRAY)
    # X Gradient，深度cv.CV_16SC1 整型
    xgrad = cv.Sobel(gray, cv.CV_16SC1, 1, 0)
    # Y Gradient
    ygrad = cv.Sobel(gray, cv.CV_16SC1, 0, 1)
    #edge
    #edge_output = cv.Canny(xgrad, ygrad, 50, 150)
    #50, 150  低阈值  高阈值
    edge_output = cv.Canny(gray, 50, 150)
    cv.imshow("Canny Edge", edge_output)
    #bitwise_and：变成彩色
    dst = cv.bitwise_and(image, image, mask=edge_output)
    cv.imshow("Color Edge", dst)


print("--------- Python OpenCV Tutorial ---------")
src = cv.imread("F:/Matlab/Asriel.jpg")
cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
cv.imshow("input image", src)
edge_demo(src)
cv.waitKey(0)

cv.destroyAllWindows()
