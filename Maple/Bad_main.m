clc; clear;
%% 读入每张图片
% for ii = 0:96
ii = 0;

Real_read = imread(['F:\Matlab\Maple\RealImg\RealImg_',num2str(ii),'.bmp']);
Ref_read = imread(['F:\Matlab\Maple\RefImg\RefImg_',num2str(ii),'.bmp']);
Real_bedge = edge(Medi_filter(Real_read, 3),'canny');
Ref_bedge = edge(Ref_read,'canny');

%% 绘制对比图
figure; 
subplot(2,2,1); imshow(Real_read)
subplot(2,2,2); imshow(Ref_read)
subplot(2,2,3); imshow(Real_bedge)
subplot(2,2,4); imshow(Ref_bedge)

Temp = Ref_read;
Temp(253,190) = 255;
subplot(2,2,2); imshow(Temp)

%% 计算中心坐标
% [Real_Lx, Real_Ly] = size(Real_read)
% [Ref_Lx, Ref_Ly] = size(Ref_read)
[Ret_X, Ret_Y] = Xor_filter(Real_bedge, Ref_bedge);




