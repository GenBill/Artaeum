clear;close all;clc;
%% �ȵ�������ѵ��ͼ��

pre = DataPreSet(pre)
train_X = pre.train_X;
test_X = pre.test_X;

%% ����5�������������δ�ڴ˶���
cnn.layers = {
    struct('type', 'i')                                         % input layer
    struct('type', 'c', 'outputmaps', 12, 'kernelsize', 5)      % convolution layer
    struct('type', 's', 'scale', 2)                             % sub-sampling layer
    struct('type', 'c', 'outputmaps', 24, 'kernelsize', 5)      % convolution layer
    struct('type', 's', 'scale', 2)                             % sub-sampling layer
};

%% ѧϰ��������
opts.alpha = 0.5;           % ѧϰ��
opts.batchsize = 25;        % ÿbatchsize��ͼ��һ��ѵ��һ�֣�����һ��Ȩֵ
opts.numepochs = 20;        % ÿ��epoch�ڣ�������ѵ�����ݽ���ѵ��

%% �����ʼ�� & ѵ��
disp('CNN Training Start ...');
cnn = cnn_setup(cnn, train_X, train_Y);
cnn = cnn_train(cnn, train_X, train_Y, opts);

%% ����������
figure; 
subplot(2,1,1); plot(cnn.rL);
title('Mean Squared Error')
subplot(2,1,2); plot(log(cnn.rL));
title('Log Mean Squared Error')

%% ������� & �ж�׼ȷ��
[test_Er, test_Bad] = cnn_test(cnn, test_X, test_Y);
disp(['10000 test images ' num2str(100-test_Er*100) '% correct']);

[train_Er, train_Bad] = cnn_test(cnn, train_X, train_Y);
disp(['60000 train images ' num2str(100-train_Er*100) '% correct']);

%% ���������
if size(test_Bad,2)<=100
    figure;
    for index = 1:size(test_Bad,2)
        subplot(10,10,index), imshow(test_X(:,:,test_Bad(index)))
    end
end
