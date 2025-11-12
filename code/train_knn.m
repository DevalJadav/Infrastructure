function model = train_knn(X,y,k)
model.X = X;
model.y = y;
model.k = k;
end

function ypred = predict_knn(model,Xtest)
n = size(Xtest,1);
ypred = zeros(n,1);
for i=1:n
    dists = sqrt(sum((model.X - Xtest(i,:)).^2,2));
    [~, idx] = sort(dists);
    nearest = model.y(idx(1:model.k));
    ypred(i) = mode(nearest);
end
end
