function visualize_results(Xtest, ytrue, ypred, classes)
figure; cm = confusionchart(ytrue, ypred);
title('Confusion Matrix of Predicted vs True Classes');
figure; gscatter(Xtest(:,1), Xtest(:,2), ypred);
xlabel('Feature 1'); ylabel('Feature 2'); title('Predicted Clusters (2D Projection)');
end
