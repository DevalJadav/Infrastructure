function feats = extract_features(img)
gray = rgb2gray(img);
edges = edge(gray,'Canny');
stats = graycoprops(graycomatrix(gray),'all');

meanColor = mean(mean(img,1),2);
stdColor = std(reshape(img,[],3));

edgesCount = sum(edges(:))/numel(edges);
entropyVal = entropy(gray);
contrastVal = stats.Contrast;
corrVal = stats.Correlation;
energyVal = stats.Energy;
homogVal = stats.Homogeneity;

feats = [meanColor(:)', stdColor(:)', edgesCount, entropyVal, contrastVal, corrVal, energyVal, homogVal];
end
