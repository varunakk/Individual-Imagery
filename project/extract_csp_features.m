function [X,Y,S,T] = extract_csp_features(EEG,Fs,mrk,wnd,f,nof,n)
fprintf('Extracting features using CSP')

% do frequency filtering using FFT
[t,c] = size(EEG); idx = reshape(1:t*c-mod(t*c,n),n,[]);
FLT = real(ifft(fft(EEG).*repmat(f(Fs*(0:t-1)/t)',1,c)));

% estimate temporal filter using least-squares
T = FLT(idx)/EEG(idx);

% extract data for all epochs of the first class concatenated (EPO{1}) and 
% all epochs of the second class concatenated (EPO{2})
% each array is [#samples x #channels]
wnd = round(Fs*wnd(1)) : round(Fs*wnd(2));

for k = 1:5
    EPO{k} = FLT(repmat(find(mrk==k),length(wnd),1) + repmat(wnd',1,nnz(mrk==k)),:);
end

% calculate the spatial filter matrix S using CSP (TODO: fill in)
C_1 = cov(EPO{1});
C_2 = cov(EPO{2});
C_3 = cov(EPO{3});
C_4 = cov(EPO{4});
C_5=cov(EPO{5});


%Form 3D cov matrix
R = zeros(5,30,30);
R(1,:,:)=C_1;
R(2,:,:)=C_2;
R(3,:,:)=C_3;
R(4,:,:)=C_4;
R(5,:,:)=C_5;


%Get projection matrix
S = MulticlassCSP(R,5*nof);

% log-variance feature extraction  If A is a matrix whose columns are random variables and whose rows are observations, then V is a row vector containing the variance corresponding to each column.

for k = 1:5
    X{k} = squeeze(log(var(reshape(EPO{k}*S', length(wnd),[],5*nof))));
end

class_1_target = ones(length(X{1}),1);
class_2_target = 2*ones(length(X{2}),1);
class_3_target = 3*ones(length(X{3}),1);
class_4_target = 4*ones(length(X{4}),1);
class_5_target = 5*ones(length(X{5}),1);


Y = vertcat(class_1_target,class_2_target,class_3_target,class_4_target,class_5_target);
X = vertcat(X{1},X{2},X{3},X{4},X{5});
end