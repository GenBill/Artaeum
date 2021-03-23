function [er, bad] = cnn_test(net, x, y)
    %  feedforward
    net = cnn_ff(net, x);
    [~, h] = max(net.o);
    [~, a] = max(y);
    bad = find(h ~= a);

    er = numel(bad) / size(y, 2);
end
