function [trainIdx, valIdx, testIdx] = stratified_split(labels_primary, tr, vr, te)
classes = unique(labels_primary);
trainIdx = []; valIdx = []; testIdx = [];
for c = classes'
    idx = find(labels_primary==c);
    n = numel(idx);
    ntr = max(1, round(tr*n)); nval = round(vr*n);
    perm = idx(randperm(n));
    trainIdx = [trainIdx; perm(1:ntr)];
    valIdx = [valIdx; perm(ntr+1:min(ntr+nval,n))];
    testIdx = [testIdx; perm(min(ntr+nval+1,n):end)];
end
end
