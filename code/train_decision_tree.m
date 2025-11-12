function model = train_decision_tree(X,y,maxDepth)
t = fitctree(X,y,'MaxNumSplits',maxDepth);
model.tree = t;
end

function ypred = predict_decision_tree(model,Xtest)
ypred = predict(model.tree,Xtest);
end
