function visualize_pipeline(images, meta, config, Xsel, Xnorm, selectedIdx, y_primary, ypred_knn, ypred_nb, ypred_dt, testIdx)
fprintf('Generating professional visualizations...\n');
if isempty(testIdx), idx = 1; else idx = testIdx(1); end
I = imread(images{idx});
numstr = regexp(images{idx}, '\d+', 'match'); numstr = numstr{1};
mask = imread(fullfile(config.maskDir, sprintf('mask%s.png', numstr)));
Iproc = preprocess_image(I, config.imgSize);
gray = rgb2gray(Iproc);
edges = edge_detect(gray);
figure('Name','Sample Processing','NumberTitle','off');
subplot(1,3,1); imshow(I); title('Original');
subplot(1,3,2); imshow(mask); title('Mask');
subplot(1,3,3); imshow(Iproc); title('Preprocessed');
feat = Xsel(idx,:);
figure('Name','Feature Snapshot','NumberTitle','off');
bar(feat); title('Feature vector (snapshot)'); xlabel('Feature index'); ylabel('Value'); grid on;
coeff = pca(Xnorm);
score = Xnorm * coeff(:,1:3);
figure('Name','PCA Scatter','NumberTitle','off');
gscatter(score(:,1), score(:,2), y_primary); xlabel('PC1'); ylabel('PC2'); title('PCA of feature space');
figure('Name','Confusion Matrices','NumberTitle','off');
subplot(1,3,1); confusionchart(y_primary(testIdx), ypred_knn); title('KNN');
subplot(1,3,2); confusionchart(y_primary(testIdx), ypred_nb); title('Naive Bayes');
subplot(1,3,3); confusionchart(y_primary(testIdx), ypred_dt); title('Decision Tree');
end
