function Img = Medi_filter(image, N)
%% 中值滤波
    [height, width] = size(image);
    x1 = double(image);
    x2 = x1;
    for i = 1: height-N+1
        for j = 1:width-N+1
            mb = x1(i:(i+N-1), j:(j+N-1));
            mb = mb(:);
            mm = median(mb);
            x2(i+(N-1)/2, j+(N-1)/2) = mm;
        end
    end
    Img = uint8(x2);
end