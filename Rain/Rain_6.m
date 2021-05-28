I = imread('GenBill.jpg');

Ir = I(:, :, 1);
Ig = I(:, :, 2);
Ib = I(:, :, 3);
gray = rgb2gray(I);

M = size(I,1);
N = size(I,2);
BW = uint8(zeros(size(I,1), size(I,2)));

for i = 1:M
    for j = 1:N
        if gray(i,j) > 240
            BW(i,j) = 0;
        else
            BW(i,j) = 1;
        end
    end
end
imshow(BW)
imLabel = bwlabel(BW);                  %对各连通域进行标记
stats = regionprops(imLabel,'Area');    %求各连通域的大小
area = cat(1,stats.Area);
index = find(area == max(area));        %求最大连通域的索引
max_BW = ismember(imLabel,index);       %获取最大连通域图像

%% 腐蚀 & 膨胀
B = [0 1 0; 1 1 1; 0 1 0];
se = strel('disk',5);
max_BW1 = imdilate(max_BW,B);
max_BW2 = imdilate(max_BW1,B);
max_BW3 = imdilate(max_BW2,B);

max_BW4 = imerode(max_BW3,se);

%% 输出图像
subplot(2,3,1);imshow(BW,[]);title('二值图像');
subplot(2,3,2);imshow(max_BW);title('最大连通分量');
subplot(2,3,3);imshow(max_BW1);title('膨胀-1');
subplot(2,3,4);imshow(max_BW2);title('膨胀-2');
subplot(2,3,5);imshow(max_BW3);title('膨胀-3');
subplot(2,3,6);imshow(max_BW4);title('腐蚀');