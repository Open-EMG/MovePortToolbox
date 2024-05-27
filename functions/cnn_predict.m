function label_predict=cnn_predict(net,feature_test,Q)

if(iscell(feature_test))
    input=[];
    for i=1:length(feature_test)
        input_tmp=reshape(feature_test{1,i},[size(feature_test{1,i},1),Q+1,1,size(feature_test{1,i},2)/(Q+1)]);
        input_tmp=permute(input_tmp,[4,2,3,1]);
        input=cat(1,input,input_tmp);
    end
else
    input=reshape(feature_test,[size(feature_test,1),Q+1,1,size(feature_test,2)/(Q+1)]);
    input=permute(input,[4,2,3,1]);
end

label_predict=predict(net,input);
[~,label_predict]=max(label_predict');
label_predict=label_predict'-1;