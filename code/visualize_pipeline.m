function visualize_pipeline(images, meta, config, Xsel, Xnorm, selectedIdx, y, ypred_knn, ypred_nb, ypred_dt, testIdx)
% Produce professional figures: sample images, preprocessing, feature bars, PCA, metrics, confusion matrices
fprintf('Generating professional visualizations...\\n');
% 1) show sample original, mask, preprocessed, edges for a representative image
idx = testIdx(1);
I = imread(images{idx}); mask = imread(fullfile(config.maskDir, sprintf('mask%s.png', regexp(images{idx}, '\d+', 'match'))));
Iproc = preprocess_image(I, config.imgSize);
gray = rgb2gray(Iproc);
edges = edge_detect(gray);

figure('Name','Sample Processing','NumberTitle','off','Units','normalized','Position',[0.05 0.55 0.4 0.35]);
subplot(1,3,1); imshow(I); title('Original');
subplot(1,3,2); imshow(mask); title('Mask');
subplot(1,3,3); imshow(Iproc); title('Preprocessed');

% 2) feature bar (first image)
feat = Xsel(idx,:);
figure('Name','Feature Snapshot','NumberTitle','off','Units','normalized','Position',[0.5 0.55 0.4 0.35]);
bar(feat); title('Feature vector (snapshot)'); xlabel('Feature index'); ylabel('Value'); grid on;

% 3) PCA scatter of normalized features colored by ground truth
coeff = pca(Xnorm);
score = Xnorm * coeff(:,1:3);
figure('Name','PCA Scatter','NumberTitle','off','Units','normalized','Position',[0.05 0.05 0.4 0.4]);
gscatter(score(:,1), score(:,2), y); xlabel('PC1'); ylabel('PC2'); title('PCA of feature space');

% 4) model metrics comparison
% compute metrics if not provided: assume metrics computed externally, but show confusion matrices
figure('Name','Confusion Matrices','NumberTitle','off','Units','normalized','Position',[0.5 0.05 0.4 0.4]);
subplot(1,3,1); confusionchart(y(testIdx), ypred_knn); title('KNN Confusion Matrix');
subplot(1,3,2); confusionchart(y(testIdx), ypred_nb); title('Naive Bayes Confusion Matrix');
subplot(1,3,3); confusionchart(y(testIdx), ypred_dt); title('Decision Tree Confusion Matrix');

end
