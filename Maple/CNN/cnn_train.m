function net = cnn_train(net, x, y, opts)
    m = size(x, 3);
    numbatches = m / opts.batchsize;
    if rem(numbatches, 1) ~= 0
        error('numbatches not integer');
    end
    net.rL = [];
    for i = 1 : opts.numepochs
        disp(['epoch ' num2str(i) '/' num2str(opts.numepochs)]);
        tic;
        kk = randperm(m);
        for j = 1 : numbatches
            batch_x = x(:, :, kk((j - 1) * opts.batchsize + 1 : j * opts.batchsize));
            batch_y = y(:,    kk((j - 1) * opts.batchsize + 1 : j * opts.batchsize));

            net = cnn_ff(net, batch_x);
            net = cnn_bp(net, batch_y);
            net = cnn_applygrads(net, opts);
            if isempty(net.rL)
                net.rL(1) = net.L;
            end
            net.rL(end + 1) = 0.99 * net.rL(end) + 0.01 * net.L;
        end
        toc;
    end
    
end
