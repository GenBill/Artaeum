function [X,Y] = Halo_rebuild(Vector, CasterLevel)
% CasterLevel = 32;
Thita_Index = [0:CasterLevel-1]'*2*pi/CasterLevel;

X = Vector.*sin(Thita_Index);
Y = Vector.*cos(Thita_Index);
plot(Y,X)

end