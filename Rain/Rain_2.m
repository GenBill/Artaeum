I = imread('GenBill.jpg');

Ir = I(:, :, 1);
Ig = I(:, :, 2);
Ib = I(:, :, 3);

%% 直方图均衡化
J = histeq(Ir);

%% 直方图规定化
Imatch = Ib;
Jmatch = imhist(Imatch);    % 获取匹配图像直方图
Iout = histeq(Ig,Jmatch);   % 直方图匹配

%% 高斯滤波
sigma = 6;                                  % 标准差大小
window = double(uint8(3*sigma)*2+1);        % 窗口大小一半为3*sigma
H = fspecial('gaussian', window, sigma);    % 产生滤波模板
img_gauss = imfilter(Ib,H,'replicate');

%% 输出图像
figure
subplot(2,3,1),imshow(Ir);title('R-原图像');
subplot(2,3,2),imshow(Ig);title('G-原图像');
subplot(2,3,3),imshow(Ib);title('B-原图像');

subplot(2,3,4),imshow(J);title('直方图均衡化');
subplot(2,3,5),imshow(Iout);title('直方图规定化');
subplot(2,3,6),imshow(img_gauss);title('高斯滤波');

