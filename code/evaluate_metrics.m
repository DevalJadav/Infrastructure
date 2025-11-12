function metrics = evaluate_metrics(ytrue, ypred, classNames)
if nargin<3, classNames = []; end
conf = confusionmat(ytrue, ypred);
accuracy = sum(diag(conf))/sum(conf(:));
nclass = size(conf,1);
precision = zeros(nclass,1); recall = zeros(nclass,1); f1 = zeros(nclass,1);
for c=1:nclass
    tp = conf(c,c); fp = sum(conf(:,c)) - tp; fn = sum(conf(c,:)) - tp;
    if tp+fp==0, precision(c)=0; else precision(c)=tp/(tp+fp); end
    if tp+fn==0, recall(c)=0; else recall(c)=tp/(tp+fn); end
    if precision(c)+recall(c)==0, f1(c)=0; else f1(c)=2*precision(c)*recall(c)/(precision(c)+recall(c)); end
end
metrics.confusion = conf; metrics.accuracy = accuracy; metrics.precision = precision; metrics.recall = recall; metrics.f1 = f1;
end
