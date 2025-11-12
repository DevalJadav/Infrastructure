function feat = extract_features(img)
% Extracts handcrafted features from preprocessed RGB image (double)
img = im2double(img);
gray = rgb2gray(img);
glcmStats = compute_glcm_stats(gray, 8);
edges = edge_detect(gray);
edgeDensity = sum(edges(:))/numel(edges);
rgb = reshape(img, [], 3);
meanColor = mean(rgb,1);
stdColor = std(rgb,0,1);
counts = imhist(gray,8); counts = counts / sum(counts);
feat = [glcmStats.contrast, glcmStats.energy, glcmStats.entropy, glcmStats.correlation, ...
        edgeDensity, meanColor, stdColor, counts(:)'];
end
