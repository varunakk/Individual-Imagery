net = patternnet(10);
Indices1=randperm(200);
y_t=Y_train(Indices1(1:160),:);
Y_test=Y_train(Indices1(161:end),:);

x_t=X_train(Indices1(1:160),:);
X_test=X_train(Indices1(161:end),:);

YANN = [(y_t == 1), (y_t == 2), (y_t == 3), (y_t == 4), (y_t == 5)];
net = train(net,transpose(x_t),transpose(YANN));
y_ann = zeros(40,1);
r=1;
for x=Indices1(161:end)      
    fprintf('\nComputation %f/%f', x, length(single(X1)))
    y_ann(r) = test_ann(single(X1(x,:)),S,T,net);
    r=r+1;
end
ll=1-mean(y_ann~=Y_test);
fprintf("loss %f\n",ll(1));
