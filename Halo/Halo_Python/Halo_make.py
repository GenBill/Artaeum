import cv2 as cv
import numpy as np

def Halo_make(Img, CasterLevel):
    xgrad = cv.Sobel(Img, cv.CV_16SC1, 1, 0)
    ygrad = cv.Sobel(Img, cv.CV_16SC1, 0, 1)
    Bedge = cv.Canny(xgrad, ygrad, 50, 150)


    Oracle = regionprops(imbinarize(Img),'centroid').Centroid
    block = imbinarize(Img)
    Bedge = edge(Img,'canny')
    # Bedge = imresize(Bedge, 10)
    # figure; imshow(Bedge)
    # BW = edge(I,'sobel')

    ## Find the Oracle
    [edge_x,edge_y] = find(Bedge==1)
    O_x = edge_x-Oracle(1)
    O_y = edge_y-Oracle(2)
    O_d = sqrt(O_y.*O_y+O_x.*O_x)

    ## cast神圣新星：128道激光
    Thita_Index = (0:CasterLevel-1).T*2*pi/CasterLevel
    for ii = 1:CasterLevel
        Falcon = sin(Thita_Index(ii)).*O_x + cos(Thita_Index(ii)).*O_y
        Falcon = Falcon .* (Falcon./O_d).^256
        
        [MaxFalcon,index] = max(Falcon)
        if MaxFalcon > 0.01
            Ret(ii,1) = O_d(index)
            Ret(ii,2) = 0
            Ret(ii,3) = index
        else
            Ret(ii,1) = 0.5
            Ret(ii,2) = 0
            Ret(ii,3) = index
        # 已使用矩阵运算加速，max函数估计为O(logn)，总时间复杂度为O(nlogn)

    ## 中值滤波
    # temp = [Ret(CasterLevel,1); Ret(:,1); Ret(1,1)]
    # Ret(:,1) = medfilt1(temp,3)
    # Ret(:,1) = medfilt1(Ret(:,1),3)

    ## 模长归一化
    SquareSum = sum(Ret(:,1).^2)
    Ret(:,1) = Ret(:,1)/SquareSum*CasterLevel

    ## 目前问题：
    # 当交点消失时，此算法会搜索一个附近的较大值来取代通常交点