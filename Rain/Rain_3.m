I = imread('GenBill.jpg');

Ir = I(:, :, 1);
Ig = I(:, :, 2);
Ib = I(:, :, 3);

M = size(I,1);
N = size(I,2);
m_mid = fix(M/2);     % 中心点坐标
n_mid = fix(N/2);

%% R通道：频率 & 相位
Ir_Fc = fftshift(fft2(Ir));
Ir_Xiang = angle(fft2(Ir));

%% G通道：高斯低通滤波
Ig_Fc = fftshift(fft2(Ig));
% Ig_Xiang = angle(fft2(Ig));

h = zeros(M,N);         % 高斯低通滤波器构造
d0 = 50;
for i = 1:M
    for j = 1:N
        d = ((i-m_mid)^2+(j-n_mid)^2);
        h(i,j) = exp(-d/(2*(d0^2)));      
    end
end

Ig_Ret = h.*Ig_Fc;

Ig_Ret = ifftshift(Ig_Ret);               % 中心平移回原来状态
Ig_Ret = uint8(real(ifft2(Ig_Ret)));      % 反傅里叶变换, 取实数部分

%% B通道：巴特沃思高通滤波
Ib_Fc = fftshift(fft2(Ib));

% 二阶巴特沃思高通滤波器，截止频率为25
nn = 2;           
d0 = 10;

Ib_Ret = zeros(M, N);
% 计算传递函数
for i = 1:M
    for j = 1:N
        d = sqrt((i-m_mid)^2+(j-n_mid)^2);
        if(d==0)
            h = 0;
        else
            h = 1/(1+0.414*(d0/d)^(2*nn));  
        end
        Ib_Ret(i,j) = h*Ib_Fc(i,j);
    end
end
Ib_Ret = ifftshift(Ib_Ret);
Ib_Ret = uint8(real(ifft2(Ib_Ret)));


%% 输出图像
figure
subplot(2,3,1),imshow(Ir_Fc);title('R-频域');
subplot(2,3,2),imshow(Ig);title('G-原图像');
subplot(2,3,3),imshow(Ib);title('B-原图像');

subplot(2,3,4);imshow(Ir_Xiang);title('R-相位');
subplot(2,3,5);imshow(Ig_Ret);title('G-高斯低通滤波 d=50');
subplot(2,3,6);imshow(Ib_Ret);title('B-巴特沃思高通滤波 d=10');
