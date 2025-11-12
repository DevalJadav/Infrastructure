function model = train_decision_tree(X,y,maxSplits)
if nargin<3, maxSplits=20; end
t = fitctree(X,y,'MaxNumSplits',maxSplits);
model.tree = t;
end
