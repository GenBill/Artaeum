I = imread('GenBill.jpg');

Ir = I(:, :, 1);
Ig = I(:, :, 2);
Ib = I(:, :, 3);

M = size(I,1);
N = size(I,2);
m_mid = fix(M/2);     % 中心点坐标
n_mid = fix(N/2);

Hsi = rgb2hsi(I);
Hsi_1 = Hsi(:, :, 1);
Hsi_2 = Hsi(:, :, 2);
Hsi_3 = Hsi(:, :, 3);

%% Fliter
New = I;
for i = 1:M
    for j = 1:N
        if abs(Hsi_1(i,j) - 0.62) < 0.1 || abs(Hsi_1(i,j) - 0.60) < 0.1
            New(i, j, 1) = 0;
            New(i, j, 2) = 0;
            New(i, j, 3) = 0;
        end
        if Hsi_3(i,j) > 0.95
            New(i, j, 1) = 0;
            New(i, j, 2) = 0;
            New(i, j, 3) = 0;
        end
    end
end

%% 彩色直方图均衡化
Ir = myHG(Ir);   %构造的函数
Ig = myHG(Ig);
Ib = myHG(Ib);
I_Ret = cat(3, Ir, Ig, Ib);  %cat用于构造多维数组

%% 输出图像
figure
subplot(2,3,1),imshow(Hsi_1);title('HSI-1');
subplot(2,3,2),imshow(Hsi_2);title('HSI-2');
subplot(2,3,3),imshow(Hsi_3);title('HSI-3');

subplot(2,3,4);imshow(New);title('面部分层');
subplot(2,3,5);imshow(I_Ret);title('彩色直方图均衡化');
