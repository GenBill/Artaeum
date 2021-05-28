I = imread('GenBill.jpg');

Ir = I(:, :, 1);
Ig = I(:, :, 2);
Ib = I(:, :, 3);

M = size(I,1);
N = size(I,2);
m_mid = fix(M/2);     % 中心点坐标
n_mid = fix(N/2);

%% 生成噪音
Noise_R = imnoise(Ir, 'gaussian');
Noise_G = imnoise(Ig, 'poisson');
Noise_B = imnoise(Ib, 'salt & pepper', 0.1);

Noise_I(:, :, 1) = Noise_R;
Noise_I(:, :, 2) = Noise_G;
Noise_I(:, :, 3) = Noise_B;

%% 恢复图像
H = fspecial('average',3);
Re_R = filter2(H, Noise_R);
Re_G = medfilt2(Noise_G);
Re_B = medfilt2(Noise_B);

Re_I(:, :, 1) = Re_R;
Re_I(:, :, 2) = Re_G;
Re_I(:, :, 3) = Re_B;
Re_I = uint8(Re_I);

%% 周期噪音
Noise_sin = zeros(M, N);
for i = 1:M
    for j = 1:N
        Noise_sin(i,j) = 20*sin(40*j);
    end
end
Ir_noise = double(Ir) + Noise_sin;

%% 周期噪音去除
freq = 725;
width = 2;
H = zeros(M,N);
for i = 1:M
    for j = 1:N
        H(i,j) = 1-exp(-0.5*((((i-M/2)^2+(j-N/2)^2)-freq^2)/(sqrt(i.^2+j.^2)*width))^2);
    end
end

Ir_Fn = fftshift(fft2(Ir_noise));
max(Ir_Fn, 10);
% imshow(log(1+abs(Ir_Fn)),[])
Ir_Ret = H.*Ir_Fn;
Re_mask = log(1+abs(Ir_Ret));

Ir_Ret = ifftshift(Ir_Ret);                 % 中心平移回原来状态
Re_Ir = uint8(real(ifft2(Ir_Ret)));         % 反傅里叶变换, 取实数部分

%% 输出图像
figure
subplot(3,3,1),imshow(Noise_R);title('R-高斯噪音');
subplot(3,3,2),imshow(Noise_G);title('G-指数噪音');
subplot(3,3,3),imshow(Noise_B);title('B-椒盐噪音');

subplot(3,3,4),imshow(Noise_I);title('RGB-噪音图像');
subplot(3,3,5),imshow(Re_I);title('RGB-恢复图像');

subplot(3,3,7),imshow(Ir_noise, []);title('R-周期噪音');
subplot(3,3,8),imshow(Re_mask, []);title('R-Mask');
subplot(3,3,9),imshow(Re_Ir, []);title('R-恢复图像');
