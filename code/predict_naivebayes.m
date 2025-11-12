function ypred = predict_naivebayes(model, Xtest)
n = size(Xtest,1); nc = numel(model.classes); ypred = zeros(n,1);
for i=1:n
    scores = zeros(1,nc);
    for c=1:nc
        mu = model.mu(c,:); v = model.sig2(c,:);
        ll = -0.5 * sum(log(2*pi*v)) - 0.5 * sum(((Xtest(i,:) - mu).^2) ./ v);
        scores(c) = log(model.prior(c)) + ll;
    end
    [~, idx] = max(scores); ypred(i) = model.classes(idx);
end
end
