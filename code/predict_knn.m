function ypred = predict_knn(model, Xtest)
n = size(Xtest,1); ypred = zeros(n,1);
for i=1:n
    d = sqrt(sum((model.X - Xtest(i,:)).^2,2));
    [~, idx] = sort(d);
    kidx = idx(1:min(model.k, numel(idx)));
    ypred(i) = mode(model.y(kidx));
end
end
