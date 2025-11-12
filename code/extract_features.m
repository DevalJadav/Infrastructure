function feat = extract_features(img)
% img assumed double [0,1], HxWx3
img = im2double(img);
gray = rgb2gray(img);
% compute GLCM stats via custom function to avoid toolbox dependency
glcmStats = compute_glcm_stats(gray, 8);
% edge density using Sobel-like gradients
edges = edge_detect(gray);
edgeDensity = sum(edges(:))/numel(edges);
% color stats
rgb = reshape(img, [], 3);
meanColor = mean(rgb,1);
stdColor = std(rgb,0,1);
% histogram features (8 bins)
counts = imhist(gray,8); counts = counts / sum(counts);
% assemble feature vector
feat = [glcmStats.contrast, glcmStats.energy, glcmStats.entropy, glcmStats.correlation, ...
        edgeDensity, meanColor, stdColor, counts(:)'];
end
