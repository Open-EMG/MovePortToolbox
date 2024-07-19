function [feature_train_norm,feature_test_norm]=feature_normalize(feature_train,feature_test)

mean_values=mean(feature_train,1);
std_values=std(feature_train,1);

std_values(find(std_values==0))=1;

feature_train_norm=(feature_train-mean_values)./std_values;
feature_test_norm=(feature_test-mean_values)./std_values;