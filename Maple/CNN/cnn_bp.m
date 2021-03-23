function net = cnn_bp(net, y)
    n = numel(net.layers);

    %  error
    net.e = net.o - y;
    %  loss function ¾ù·½Îó²î
    net.L = 1/2 * sum(net.e(:) .^ 2) / size(net.e, 2);

    %%  backprop deltas
    net.od = net.e .* (net.o .* (1 - net.o));   %  output delta
    net.fvd = (net.ffW' * net.od);              %  feature vector delta
    if strcmp(net.layers{n}.type, 'c')          %  only conv layers has sigm function
        net.fvd = net.fvd .* (net.fv .* (1 - net.fv));
    end

    %  reshape feature vector deltas into output map style
    sa = size(net.layers{n}.a{1});
    fvnum = sa(1) * sa(2);
    for j = 1 : numel(net.layers{n}.a)
        net.layers{n}.d{j} = reshape(net.fvd(((j - 1) * fvnum + 1) : j * fvnum, :), sa(1), sa(2), sa(3));
    end

    for Li = (n - 1) : -1 : 1
        if strcmp(net.layers{Li}.type, 'c')
            for j = 1 : numel(net.layers{Li}.a)
                net.layers{Li}.d{j} = net.layers{Li}.a{j} .* (1 - net.layers{Li}.a{j}) .* (expand(net.layers{Li + 1}.d{j}, [net.layers{Li + 1}.scale net.layers{Li + 1}.scale 1]) / net.layers{Li + 1}.scale ^ 2);
            end
        elseif strcmp(net.layers{Li}.type, 's')
            for i = 1 : numel(net.layers{Li}.a)
                z = zeros(size(net.layers{Li}.a{1}));
                for j = 1 : numel(net.layers{Li + 1}.a)
                     z = z + convn(net.layers{Li + 1}.d{j}, rot180(net.layers{Li + 1}.k{i}{j}), 'full');
                end
                net.layers{Li}.d{i} = z;
            end
        end
    end

    %%  calc gradients
    for Li = 2 : n
        if strcmp(net.layers{Li}.type, 'c')
            for j = 1 : numel(net.layers{Li}.a)
                for i = 1 : numel(net.layers{Li - 1}.a)
                    net.layers{Li}.dk{i}{j} = convn(flipall(net.layers{Li - 1}.a{i}), net.layers{Li}.d{j}, 'valid') / size(net.layers{Li}.d{j}, 3);
                end
                net.layers{Li}.db{j} = sum(net.layers{Li}.d{j}(:)) / size(net.layers{Li}.d{j}, 3);
            end
        end
    end
    net.dffW = net.od * (net.fv)' / size(net.od, 2);
    net.dffb = mean(net.od, 2);

    function X = rot180(X)
        X = flip(flip(X, 1), 2);
    end
end
