I = imread('GenBill.jpg');

Ir = I(:, :, 1);
Ig = I(:, :, 2);
Ib = I(:, :, 3);
I0 = Ib.*0;

figure
subplot(2, 3, 1);   imshow(cat(3, cat(3, Ir, I0), I0))
subplot(2, 3, 2);   imshow(cat(3, cat(3, I0, Ig), I0))
subplot(2, 3, 3);   imshow(cat(3, cat(3, I0, I0), Ib))

max(Ir(:))
min(Ig(:))
mean(Ib(:))

subplot(2, 3, 4);   histogram(Ir)
subplot(2, 3, 5);   histogram(Ig)
subplot(2, 3, 6);   histogram(Ib)
