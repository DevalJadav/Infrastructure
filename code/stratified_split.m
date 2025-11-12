function [trainIdx, valIdx, testIdx] = stratified_split(labels, tr, vr, te)
classes = unique(labels);
trainIdx = []; valIdx = []; testIdx = [];
for c = classes'
    idx = find(labels==c);
    n = numel(idx);
    ntr = round(tr*n); nval = round(vr*n);
    perm = randperm(n);
    trainIdx = [trainIdx; idx(perm(1:ntr))];
    valIdx = [valIdx; idx(perm(ntr+1:ntr+nval))];
    testIdx = [testIdx; idx(perm(ntr+nval+1:end))];
end
end
