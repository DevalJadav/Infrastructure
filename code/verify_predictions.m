function verify_predictions(images, maskDir, testIdx, knnModel, nbModel, dtModel, cfg, numSamples)
fprintf('Verifying predictions on random samples...\n');
load(fullfile('results','features.mat'),'selected','mu','sigma');
randIdx = randsample(testIdx, min(numSamples, numel(testIdx)));
figure('Name','Model Predictions vs Ground Truth','Color','w');
for i = 1:numel(randIdx)
    idx = randIdx(i);
    I = imread(images{idx});
    Iproc = preprocess_image(I, cfg.imgSize);
    feat = extract_features(Iproc);
    feat = feat(:, selected);
    featn = (feat - mu) ./ (sigma + eps);
    pred_knn = predict_knn(knnModel, featn);
    pred_nb  = predict_naivebayes(nbModel, featn);
    pred_dt  = predict_decision_tree(dtModel, featn);
    numstr = regexp(images{idx}, '\d+', 'match', 'once');
    maskPath = fullfile(maskDir, sprintf('mask%s.png', numstr));
    mask = imread(maskPath);
    subplot(numel(randIdx), 3, (i-1)*3+1);
    imshow(I); title(sprintf('Image #%s', numstr));
    subplot(numel(randIdx), 3, (i-1)*3+2);
    imshow(mask); title('Ground Truth Mask');
    subplot(numel(randIdx), 3, (i-1)*3+3);
    bar([pred_knn pred_nb pred_dt]);
    title('Model Predictions');
    set(gca,'XTickLabel',cfg.classes,'XTickLabelRotation',30);
end
end
