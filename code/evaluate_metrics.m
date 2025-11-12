function metrics = evaluate_metrics(ytrue, ypred, classes)
confMat = confusionmat(ytrue, ypred);
accuracy = sum(diag(confMat)) / sum(confMat(:));
precision = diag(confMat) ./ sum(confMat,2);
recall = diag(confMat) ./ sum(confMat,1)';
F1 = 2*(precision.*recall)./(precision+recall);
metrics.confMat = confMat;
metrics.accuracy = accuracy;
metrics.precision = precision;
metrics.recall = recall;
metrics.F1 = F1;
end
