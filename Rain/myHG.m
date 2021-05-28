function d = myHG(I)%构造histogram函数
J=I;
[m,n]=size(I);      %确定矩阵大小
area=m*n;
a=zeros(1,256);     %产生1*256的零矩阵a,用来存放原始图像各个灰度值的个数
b=zeros(1,256);
for i=1:m           %记录各个灰度值的个数
    for j=1:n
        d=I(i,j)+1;   %获取(i,j)位置的灰度值(注意：灰度值为0-255，对应矩阵的1-256）
        a(1,d)=a(1,d)+1;    %矩阵a上对应灰度值的计数+1
    end
end
for i=1:256         %均衡化
    sum=0;
    for j=1:i
        sum=sum+a(1,j);
    end
    b(1,i)=sum*255/area;
end
for i=1:m           %用均衡化后的数据代替原位置的数据
    for j=1:n
        d=J(i,j)+1;
        J(i,j)=b(1,d);
    end
end
d=J;