v =[7 10 15 20 25];
for i = 1:length(v)
    nor(i) = (v(i) - min(v))/(max(v)-min(v));
end
disp(nor)

A =[7 10 15 20 25];
rata2= mean(A);
c = 0;
for i = 1:length(A)
    d(i) = (A(i)-rata2)^2;
    c = c+d(i)
    sd = sqrt(c/length(A));
end
fprintf('Standar Deviasi = %.4f/n Data Baru = ', sd);
for i = 1:length(A)
    x(i)= (A(i)-rata2)/sd;
end
disp(x)

M = readtable("shopping_data.xlsx")
Normalisasi1 = normalize(A);
Normalisasi2 = normalize(A, 'zscore');
Normalisasi3 = normalize(A, 'scale');
Normalisasi4 = normalize(A, 'range');


