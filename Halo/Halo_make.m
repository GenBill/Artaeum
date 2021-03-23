function Ret = Halo_make(Img, CasterLevel)
Ret = [];
Oracle = round(regionprops(imbinarize(Img),'centroid').Centroid);
block = imbinarize(Img);
Bedge = edge(Img,'canny');
% Bedge = imresize(Bedge, 10);
figure; imshow(Bedge)
% BW = edge(I,'sobel');

%% Find the Oracle
Img(Oracle(1),Oracle(2)) = 0;
[edge_x,edge_y] = find(Bedge==1);
O_x = edge_x-Oracle(1);
O_y = edge_y-Oracle(2);
O_d = sqrt(O_y.*O_y+O_x.*O_x);

%% cast神圣新星：128道激光
% CasterLevel = 32;
Thita_Index = [0:CasterLevel-1]'*2*pi/CasterLevel;
temp1 = cos(Thita_Index);   % degree: 128
temp2 = O_y./O_d;           % degree: 93

% 首先区分y的正负性
[Y_posIndex, Y_pos] = find(O_y>=0);
[Y_negIndex, Y_neg] = find(O_y<0);
[temp_pos, sortposIndex] = sortrows(temp2(Y_posIndex),-1);
[temp_neg, sortnegIndex] = sortrows(temp2(Y_negIndex),-1);

%% 算法思路
% 先将图形放大（插值平滑曲线），再寻找最外层边缘
% 按照这个算法，小图是找不到最外层边缘的
% 此算法经过类人认证！！！

% 点积平方/模长
% for 128条Aegis Nova射线，遍历有序cos序列，寻找相交点

jj = 1;
for ii = 1:(CasterLevel/2)
    Ret(ii,2) = inf;
    % find(角度接近&&y>=0)
    for jj = 1:size(temp_pos,1)
        if(abs(temp1(ii)-temp_pos(jj)) < Ret(ii,2))
            Ret(ii,1) = O_d(sortposIndex(jj));
            Ret(ii,2) = abs(temp1(ii)-temp_pos(jj));
            Ret(ii,3) = sortposIndex(jj);
        end
    end
    % O(n^2)时间复杂度，需要后续优化为O(n)
end

jj = 1;
for ii = (CasterLevel/2+1):CasterLevel
    Ret(ii,2) = inf;
    % find(角度接近&&y>=0)
    for jj = 1:size(temp_neg,1)
        if(abs(temp1(ii)-temp_neg(jj)) < Ret(ii,2))
            Ret(ii,1) = O_d(sortnegIndex(jj));
            Ret(ii,2) = abs(temp1(ii)-temp_neg(jj));
            Ret(ii,3) = sortnegIndex(jj);
        end
    end
    % O(n^2)时间复杂度，需要后续优化为O(n)
end

end