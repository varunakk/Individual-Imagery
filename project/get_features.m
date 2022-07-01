a=load("A.mat"); 
data = a.data;
struct1 = data{1,1};
X1 = struct1.X;
y1 = struct1.y;
trail1 = struct1.trial;
fs1 = struct1.fs;
classes1 = struct1.classes;

flt = @(f)(f>7&f<30).*(1-cos((f-(7+30)/2)/(7-30)*pi*4));
  
[X_train,Y_train, S, T] = extract_csp_features(single(X1), fs1, sparse(1,trail1,(y1)),[0.5 3.5],flt,3,200);
save('features.mat', 'X_train')
save('targets.mat', 'Y_train')
save('spatial_transform.mat', 'S')
save('temporal_transform.mat', 'T')
