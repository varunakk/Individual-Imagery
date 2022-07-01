function W = MulticlassCSP(R,N);

% Add ffdiag to path
path(path,'ffdiag_pack');

% Parameters
Classes = size(R,1); % number of classes
Chans = size(R,2); % number of channels
Pc = ones(1,Classes)./Classes; % uniform prior over classes

% Jointly diagonalize covariance matrices
disp('Jointly diagonalizing covariance matrices...');
[V,CD,stat] = ffdiag(shiftdim(R,1),eye(Chans));
V = V';

% Compute mutual information provided by each filter (approximation)
disp('Selecting spatial filters with maximum mutual information...');
for n1 = 1:1:Chans,
    w = V(:,n1);
    I(n1) = J_ApproxMI(w,R,Pc);
end
[dummy(n1,:) iMI] = sort(I,'descend');
W = V(:,iMI(1:N))';

disp('Done.');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function I = J_ApproxMI(w,R,Pc)

% Set dimensions
M = size(R,1); % number of classes
N = size(R,2); % data dimension

% Compute marginalized covariance matrix
Rx = zeros(N,N);
for n1 = 1:1:M,
    Rx = Rx + Pc(n1)*reshape(R(n1,:,:),N,N);
end

% Normalize variance of x
wv = w'*Rx*w;
w = w./sqrt(wv);

% Compute gaussian mutual information
Ig = 1/2*log((w'*Rx*w));
for n1 = 1:1:M,
    Ig = Ig - 1/2*Pc(n1)*log(w'*reshape(R(n1,:,:),N,N)*w);
end

% Compute estimate of negentropy
J = 0;
for n1 = 1:1:M,
    J = J + Pc(n1)*(w'*reshape(R(n1,:,:),N,N)*w)^2;
end
J = (J - 1)^2;
J = 9/48*J;

% Compute estimate of mutual information with correction by negentropy
I = Ig - J;