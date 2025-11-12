function verify_dataset_balance(labels_multi, classNames)
counts = sum(labels_multi,1);
figure('Name','Class Distribution','NumberTitle','off');
bar(counts);
set(gca,'xticklabel',classNames);
ylabel('Number of images (may include multi-labels)');
title('Dataset class distribution (multi-label)');
for i=1:numel(counts), text(i, counts(i)+2, num2str(counts(i)), 'HorizontalAlignment','center'); end
end
