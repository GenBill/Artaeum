% CasterLevel = 32;
CasterLevel = 128;
Length = 256;
Xe = zeros(CasterLevel,Length);
Ye = zeros(CasterLevel,Length);

for ii = 0:CasterLevel-1
    thisAngle = ii*pi/CasterLevel*2;
    thisTan = tan(thisAngle);
    if abs(thisTan)<=1
        for jj = 1:(2*Length+1)
            index = jj-1-Length;
            Xe(ii+1,jj) = index;
            Ye(ii+1,jj) = round(index*thisTan);
        end
    else
        for jj = 1:(2*Length+1)
            index = jj-1-Length;
            Xe(ii+1,jj) = round(index/thisTan);
            Ye(ii+1,jj) = index;
        end
    end
end

hold on
axis equal
for ii = 1:CasterLevel
    plot(Xe(ii,:),Ye(ii,:))
end
