clear;close all;clc;
%% 先导入所有训练图像

pre = DataPreSet(pre)
train_X = pre.train_X;
test_X = pre.test_X;

%% 定义5层神经网，输出层未在此定义
cnn.layers = {
    struct('type', 'i')                                         % input layer
    struct('type', 'c', 'outputmaps', 12, 'kernelsize', 5)      % convolution layer
    struct('type', 's', 'scale', 2)                             % sub-sampling layer
    struct('type', 'c', 'outputmaps', 24, 'kernelsize', 5)      % convolution layer
    struct('type', 's', 'scale', 2)                             % sub-sampling layer
};

%% 学习参数设置
opts.alpha = 0.5;           % 学习率
opts.batchsize = 25;        % 每batchsize张图像一起训练一轮，调整一次权值
opts.numepochs = 20;        % 每个epoch内，对所有训练数据进行训练

%% 网络初始化 & 训练
disp('CNN Training Start ...');
cnn = cnn_setup(cnn, train_X, train_Y);
cnn = cnn_train(cnn, train_X, train_Y, opts);

%% 输出均方误差
figure; 
subplot(2,1,1); plot(cnn.rL);
title('Mean Squared Error')
subplot(2,1,2); plot(log(cnn.rL));
title('Log Mean Squared Error')

%% 网络测试 & 判断准确率
[test_Er, test_Bad] = cnn_test(cnn, test_X, test_Y);
disp(['10000 test images ' num2str(100-test_Er*100) '% correct']);

[train_Er, train_Bad] = cnn_test(cnn, train_X, train_Y);
disp(['60000 train images ' num2str(100-train_Er*100) '% correct']);

%% 输出错误案例
if size(test_Bad,2)<=100
    figure;
    for index = 1:size(test_Bad,2)
        subplot(10,10,index), imshow(test_X(:,:,test_Bad(index)))
    end
end
