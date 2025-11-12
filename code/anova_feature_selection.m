function pValues = anova_feature_selection(X, y)
% Handles multi-label (y can be NxC) or single-label (Nx1)
if size(y,2) > 1
    [~, y_primary] = max(y, [], 2);
else
    y_primary = y;
end
nFeatures = size(X,2);
pValues = ones(1,nFeatures);
for f=1:nFeatures
    feature = X(:,f);
    if any(isnan(feature)) || any(isinf(feature))
        feature = fillmissing(feature,'constant',0);
    end
    try
        p = anova1(feature, y_primary, 'off');
    catch
        p = 1;
    end
    pValues(f) = p;
end
fprintf('ANOVA complete: %d / %d features significant (p <= 0.05)\n', sum(pValues<=0.05), nFeatures);
end
