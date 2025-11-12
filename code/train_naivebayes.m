function model = train_naivebayes(X,y)
classes = unique(y);
nClass = numel(classes);
[n, m] = size(X);
mu = zeros(nClass,m); sigma = zeros(nClass,m);
prior = zeros(nClass,1);
for i=1:nClass
    Xi = X(y==classes(i),:);
    mu(i,:) = mean(Xi,1);
    sigma(i,:) = std(Xi,[],1)+1e-6;
    prior(i) = size(Xi,1)/n;
end
model.mu = mu; model.sigma = sigma; model.prior = prior; model.classes = classes;
end

function ypred = predict_naivebayes(model,Xtest)
n = size(Xtest,1);
ypred = zeros(n,1);
for i=1:n
    probs = zeros(1,numel(model.classes));
    for c=1:numel(model.classes)
        p = log(model.prior(c)) - 0.5*sum(((Xtest(i,:) - model.mu(c,:)).^2)./(model.sigma(c,:).^2));
        probs(c) = p;
    end
    [~, idx] = max(probs);
    ypred(i) = model.classes(idx);
end
end
