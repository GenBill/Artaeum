function Halo_show(testID, CasterLevel)
%% 先导入所有训练图像
% numTrain = 1000;
% numTest = 100;
X = loadMNISTImages('train-images-idx3-ubyte')';
Y = loadMNISTLabels('train-labels-idx1-ubyte');
Xtest = loadMNISTImages('t10k-images-idx3-ubyte')';
Ytest = loadMNISTLabels('t10k-labels-idx1-ubyte');

%% 膨胀核
% B = [0 1 0;
%     1 1 1;
%     0 1 0];
% A2 = imdilate(A1,B);    %图像A1被结构元素B膨胀

%% 数据测试部分
% Img = reshape(X(ii,:),28,28);
Img = reshape(Xtest(testID,:),28,28);
figure; subplot(1,2,1); imshow(Img)
Img = flipud(Img);
% Img = imresize(Img, 10);
% this_Halo = Halo_make(Img,CasterLevel);
this_Halo = Ring_make(Img,CasterLevel);
this_Ring = this_Halo(:,1);
subplot(1,2,2); Halo_rebuild(this_Ring,CasterLevel);