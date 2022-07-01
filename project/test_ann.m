function [ y ] = test_ann(X, S, T, model)
global B;
if any(size(B) ~= [length(T),length(S)])
    B = zeros(length(T),length(S));
end
B = [B;X]; B = B(end-length(T)+1:end,:);

X = log(var(T*(B*S')));
tmp = model(transpose(X));
[h, i] = max(tmp);
fprintf("%f  ",i);
%if i == 4
%y = 4;
%else 
%y = 1;
%end
y=i;
end
