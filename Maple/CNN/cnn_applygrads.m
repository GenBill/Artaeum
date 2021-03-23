function net = cnn_applygrads(net, opts)
    for Li = 2 : numel(net.layers)
        if strcmp(net.layers{Li}.type, 'c')
            for j = 1 : numel(net.layers{Li}.a)
                for i = 1 : numel(net.layers{Li - 1}.a)
                    net.layers{Li}.k{i}{j} = net.layers{Li}.k{i}{j} - opts.alpha * net.layers{Li}.dk{i}{j};
                end
                net.layers{Li}.b{j} = net.layers{Li}.b{j} - opts.alpha * net.layers{Li}.db{j};
            end
        end
    end

    net.ffW = net.ffW - opts.alpha * net.dffW;
    net.ffb = net.ffb - opts.alpha * net.dffb;
end
