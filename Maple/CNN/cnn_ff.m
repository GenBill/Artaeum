function net = cnn_ff(net, x)
    n = numel(net.layers);
    net.layers{1}.a{1} = x;
    inputmaps = 1;
    % 创建函数句柄加速
    sigm = @sigmoid;
    for Li = 2 : n   %  for each layer
        if strcmp(net.layers{Li}.type, 'c')
            %  !!below can probably be handled by insane matrix operations
            for j = 1 : net.layers{Li}.outputmaps   %  for each output map
                %  create temp output map
                %  初始化 z 为零矩阵
                z = zeros(size(net.layers{Li - 1}.a{1}) - [net.layers{Li}.kernelsize - 1 net.layers{Li}.kernelsize - 1 0]);
                for i = 1 : inputmaps   %  for each input map
                    %  convolve with corresponding kernel and add to temp output map
                    z = z + convn(net.layers{Li - 1}.a{i}, net.layers{Li}.k{i}{j}, 'valid');
                end
                %  add bias, pass through nonlinearity
                net.layers{Li}.a{j} = sigm(z + net.layers{Li}.b{j});
                %  将计算结果赋值给下一层
            end
            %  set number of input maps to this layers number of outputmaps
            inputmaps = net.layers{Li}.outputmaps;
        elseif strcmp(net.layers{Li}.type, 's')
            %  池化/下采样 downsample
            for j = 1 : inputmaps
                z = convn(net.layers{Li - 1}.a{j}, ones(net.layers{Li}.scale) / (net.layers{Li}.scale ^ 2), 'valid');   %  !! replace with variable
                net.layers{Li}.a{j} = z(1 : net.layers{Li}.scale : end, 1 : net.layers{Li}.scale : end, :);
            end
        end
    end

    %  concatenate all end layer feature maps into vector
    %  将最后一层的结果以向量形式输出
    net.fv = [];
    for j = 1 : numel(net.layers{n}.a)
        sa = size(net.layers{n}.a{j});
        net.fv = [net.fv; reshape(net.layers{n}.a{j}, sa(1) * sa(2), sa(3))];
    end
    %  feedforward into output perceptrons
    %  最终输出是一个10维向量，分别代表图像指向0:9的概率值
    net.o = sigm(net.ffW * net.fv + repmat(net.ffb, 1, size(net.fv, 2)));
    %  Output = [sum(ffW*fv) + ffb] (0:9)
end
