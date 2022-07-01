Indices1=randperm(200);
y_t=Y_train(Indices1(1:160),:);
Y_test=Y_train(Indices1(161:end),:);

x_t=X_train(Indices1(1:160),:);
X_test=X_train(Indices1(161:end),:);


random_forest = fitcensemble(x_t, y_t,'Method', 'Bag');
y_random_forest = zeros(size(Y_test));

r=1;
for x=Indices1(161:end)
    y_random_forest(r) = test_svm(single(X1(x,:)),S,T,random_forest);
    r=r+1;
end

loss = 1-mean(y_random_forest~=Y_test);
fprintf('The Random Forest mis-classification rate on the test set is %.2f \n',loss);
