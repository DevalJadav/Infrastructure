function model = train_naivebayes(X,y)
classes = unique(y); nc = numel(classes);
[n,m] = size(X);
mu = zeros(nc,m); sig2 = zeros(nc,m); prior = zeros(nc,1);
for i=1:nc
    Xi = X(y==classes(i),:);
    mu(i,:) = mean(Xi,1);
    sig2(i,:) = var(Xi,0,1) + 1e-6;
    prior(i) = size(Xi,1)/n;
end
model.mu = mu; model.sig2 = sig2; model.prior = prior; model.classes = classes;
end
