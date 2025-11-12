function models = train_classifiers(X, Y)
% TRAIN_CLASSIFIERS trains multiple simple classifiers.

models.knn = fitcknn(X, Y, 'NumNeighbors', 3);
models.nb  = fitcnb(X, Y);
models.tree = fitctree(X, Y);
end
