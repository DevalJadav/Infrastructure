function pValues = anova_feature_selection(X,y)
[n, m] = size(X);
pValues = zeros(1,m);
for j=1:m
    groups = cell(1,max(y));
    for c=1:max(y)
        groups{c} = X(y==c,j);
    end
    p = anova1(X(:,j), y, 'off');
    pValues(j) = p;
end
end
