function stats = compute_glcm_stats(gray, nLevels)
% Compute simple averaged GLCM stats for 4 offsets
if nargin<2, nLevels = 8; end
I = im2uint8(gray);
% quantize
Iq = floor(double(I)/256 * nLevels) + 1;
H = size(Iq,1); W = size(Iq,2);
offsets = [0 1; -1 1; -1 0; -1 -1];
allContrast = zeros(1,size(offsets,1));
allEnergy = zeros(1,size(offsets,1));
allEntropy = zeros(1,size(offsets,1));
allCorr = zeros(1,size(offsets,1));

for k=1:size(offsets,1)
    dx = offsets(k,1); dy = offsets(k,2);
    P = zeros(nLevels,nLevels);
    for x=1:H
        for y=1:W
            x2 = x + dx; y2 = y + dy;
            if x2>=1 && x2<=H && y2>=1 && y2<=W
                i = Iq(x,y); j = Iq(x2,y2);
                P(i,j) = P(i,j) + 1;
            end
        end
    end
    if sum(P(:))==0, Pn = P; else Pn = P/sum(P(:)); end
    [Igrid,Jgrid] = meshgrid(1:nLevels,1:nLevels);
    allContrast(k) = sum((Igrid(:)-Jgrid(:)).^2 .* Pn(:));
    allEnergy(k) = sum(Pn(:).^2);
    pnz = Pn(Pn>0);
    allEntropy(k) = -sum(pnz .* log2(pnz));
    mu_i = sum(Igrid(:).*Pn(:)); mu_j = sum(Jgrid(:).*Pn(:));
    sigma_i = sqrt(sum((Igrid(:)-mu_i).^2 .* Pn(:))); sigma_j = sqrt(sum((Jgrid(:)-mu_j).^2 .* Pn(:)));
    if sigma_i*sigma_j==0, allCorr(k)=0; else allCorr(k)= sum(((Igrid(:)-mu_i).*(Jgrid(:)-mu_j)).*Pn(:)) / (sigma_i*sigma_j); end
end
stats.contrast = mean(allContrast);
stats.energy = mean(allEnergy);
stats.entropy = mean(allEntropy);
stats.correlation = mean(allCorr);
end
