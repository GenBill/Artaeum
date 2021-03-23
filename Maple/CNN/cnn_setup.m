function net = cnn_setup(net, x, y)
    inputmaps = 1;
    mapsize = size(squeeze(x(:, :, 1)));
    %% ��ÿһ��������б�������
    for Li = 1 : numel(net.layers)
        %% �ػ��㴦����ȡ�ػ����mapsize���ͳ�ʼƫ��
        if strcmp(net.layers{Li}.type, 's')
            mapsize = mapsize / net.layers{Li}.scale;
            assert(all(floor(mapsize)==mapsize), ['Layer ' num2str(Li) ' size must be integer. Actual: ' num2str(mapsize)]);
            for j = 1 : inputmaps
                net.layers{Li}.b{j} = 0;
            end
        end
        %% ����㴦����ȡ��һ���mapsize��Ȩ�غ�ƫ�õĳ�ʼ������һ���inputmaps
        if strcmp(net.layers{Li}.type, 'c')
            mapsize = mapsize - net.layers{Li}.kernelsize + 1;
            fan_out = net.layers{Li}.outputmaps * net.layers{Li}.kernelsize ^ 2;
            for j = 1 : net.layers{Li}.outputmaps   % output map
                fan_in = inputmaps * net.layers{Li}.kernelsize ^ 2;
                for i = 1 : inputmaps               % input map
                    net.layers{Li}.k{i}{j} = (rand(net.layers{Li}.kernelsize) - 0.5) * 2 * sqrt(6 / (fan_in + fan_out));
                end
                net.layers{Li}.b{j} = 0;
            end
            inputmaps = net.layers{Li}.outputmaps;
        end
    end
    %% ��ʼ��������������
    % 'onum' is the number of labels, that's why it is calculated using size(y, 1). If you have 20 labels so the output of the network will be 20 neurons.
    % 'fvnum' is the number of output neurons at the last layer, the layer just before the output layer.
    % 'ffb' is the biases of the output neurons.
    % 'ffW' is the weights between the last layer and the output neurons. Note that the last layer is fully connected to the output layer, that's why the size of the weights is (onum * fvnum)
    fvnum = prod(mapsize) * inputmaps;
    onum = size(y, 1);

    net.ffb = zeros(onum, 1);
    net.ffW = (rand(onum, fvnum) - 0.5) * 2 * sqrt(6 / (onum + fvnum));
end
