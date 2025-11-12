function model = train_decision_tree(X,y,maxSplits)
% Use fitctree if available for reliable tree implementation
if nargin<3, maxSplits = 20; end
t = fitctree(X, y, 'MaxNumSplits', maxSplits);
model.tree = t;
end
