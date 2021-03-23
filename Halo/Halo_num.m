clear;clc;
tic
%% 先导入所有训练图像
numTrain = 100;
numTest = 10000;
X = loadMNISTImages('train-images-idx3-ubyte')';
Y = loadMNISTLabels('train-labels-idx1-ubyte');
Xtest = loadMNISTImages('t10k-images-idx3-ubyte')';
Ytest = loadMNISTLabels('t10k-labels-idx1-ubyte');

%% 膨胀核
% B = [0 1 0;
%     1 1 1;
%     0 1 0];
% A2 = imdilate(A1,B);    %图像A1被结构元素B膨胀
CasterLevel = 32;

%% 提取边缘
All_Ring = zeros(CasterLevel,numTrain);
for ii = 1:numTrain
    Img = reshape(X(ii,:),28,28);
    % Img = imresize(Img, 10);
    % this_Halo = Halo_make(Img,CasterLevel);
    this_Halo = Ring_make(Img,CasterLevel);
    All_Ring(:,ii) = this_Halo(:,1)';
end
toc; tic
%% KNN - Test
accu = 0;
for ii = 1:numTest
    all_Dist = ones(numTest)*inf;
    this_Img = reshape(X(ii,:),28,28);
    this_Halo = Ring_make(this_Img,CasterLevel);
    this_Ring = this_Halo(:,1);
    
    for jj = 1:numTrain
        all_Dist(jj) = Halo_distance(All_Ring(:,jj),this_Ring);
    end
    [new_Dist, Index] = sort(all_Dist);
    class = Y(Index(1:10));
    class = [Ytest(ii);class];
    MatClass(:,ii) = class;
    if class(1)~=class(2) && class(2)~=class(3)
        Reclass=class(1);
    elseif class(1)==class(2)
        Reclass=class(1);
	elseif class(2)==class(3)
        Reclass=class(2);
    end
	
	if Reclass == Ytest(ii)        
        accu = accu+1;
	end
end

%% 结论：输出识别率
toc
accuracy = accu/numTest

%% Halo_compare
% isEqual = zeros(1000,1000);
% Halo_Dist = zeros(1000,1000);
% for ii = 1:1000
%     for jj = ii:1000
%         isEqual(ii,jj) = Ytest(ii)==Ytest(jj);
%         isEqual(jj,ii) = Ytest(ii)==Ytest(jj);
%         Halo_Dist(ii,jj) = Halo_compare(All_Ring(:,ii),All_Ring(:,jj));
%         Halo_Dist(jj,ii) = Halo_Dist(ii,jj);
%     end
% end
% Good = Halo_Dist(find(isEqual==1)); hist(Good)
% Bad = size(find(isEqual(find(Halo_Dist<100))==0),1)/1000
