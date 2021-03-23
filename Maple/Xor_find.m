function [Ret_X, Ret_Y] = Xor_find(Real_bedge, Ref_bedge)
% 返回匹配区左上角的坐标
Real_bedge = double(Real_bedge);
Ref_bedge = double(Ref_bedge);

[Real_Lx, Real_Ly] = size(Real_bedge);
[Ref_Lx, Ref_Ly] = size(Ref_bedge);

Ret_X = 1;  Ret_Y = 1;
SumXor = zeros(Ref_Lx-Real_Lx,Ref_Ly-Real_Ly);
MaXor = 0;
for ii = 1:Ref_Lx-Real_Lx
    for jj = 1:Ref_Ly-Real_Ly
        Xor_Mat = zeros(Real_Lx,Real_Ly);
        % SumXor = 0;
        % XorMat = xor(Real_bedge,Ref_bedge(ii:ii+Real_Lx-1,jj:jj+Real_Ly-1));
        Xor_temp1 = imdilate(Ref_bedge(ii:ii+Real_Lx-1,jj:jj+Real_Ly-1),ones(3,3));
        Xor_temp2 = Ref_bedge(ii:ii+Real_Lx-1,jj:jj+Real_Ly-1);
        Xor_Mat = 2 - xor(Real_bedge,Xor_temp1) - xor(Real_bedge,Xor_temp2);
        % Xor_Mat = 1 - xor(Real_bedge,Xor_temp1);
        SumXor(ii,jj) = sum(sum(Xor_Mat));
        
        if (SumXor(ii,jj) > MaXor)
            Ret_X = ii;
            Ret_Y = jj;
            MaXor = SumXor(ii,jj);
        end
    end
end


[Ret_X, Ret_Y]
SumXor = SumXor/(Real_Lx*Real_Ly)*256;
imshow(SumXor,[])


