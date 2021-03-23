function Ret = Halo_distance(Halo_1, Halo_2)
% Halo_1 = rand(32,1);
% Halo_2 = rand(32,1);
CasterLevel = size(Halo_1,1);
maxAngle = round(CasterLevel/8);
Halo_3 = zeros(CasterLevel,2*maxAngle+1);

for ii = 1:maxAngle
    Halo_temp = [Halo_2(ii+1:CasterLevel); Halo_2(1:ii)];
    Halo_3(:,ii) = Halo_temp;
end

for ii = 1:maxAngle
    Halo_temp = [Halo_2(CasterLevel-ii+1:CasterLevel); Halo_2(1:CasterLevel-ii)];
    Halo_3(:,ii+maxAngle) = Halo_temp;
end

Halo_3(:,2*maxAngle+1) = Halo_2;
eHalo = (Halo_3-Halo_1).^16;
Ret = min(sum(eHalo));

% eHalo = Halo_3.*Halo_1;
% Ret = max(sum(eHalo));

end
