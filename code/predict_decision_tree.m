function ypred = predict_decision_tree(model, Xtest)
ypred = predict(model.tree, Xtest);
end
