function metrics = evaluate_metrics(ytrue, ypred, classNames)
conf = confusionmat(ytrue, ypred);
accuracy = sum(diag(conf))/sum(conf(:));
nclass = size(conf,1);
precision = zeros(nclass,1); recall=zeros(nclass,1); f1=zeros(nclass,1);
for c=1:nclass
    tp = conf(c,c); fp = sum(conf(:,c)) - tp; fn = sum(conf(c,:)) - tp;
    precision(c) = tp/(tp+fp+eps); recall(c)=tp/(tp+fn+eps);
    f1(c) = 2*precision(c)*recall(c)/(precision(c)+recall(c)+eps);
end
metrics.conf = conf; metrics.accuracy=accuracy; metrics.precision=precision; metrics.recall=recall; metrics.f1=f1;
end
