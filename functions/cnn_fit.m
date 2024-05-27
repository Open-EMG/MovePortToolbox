function net=cnn_fit(feature_train,label_train,Q,miniBatchSize)

idx=randperm(length(label_train));
idx_validation=idx(1:round(length(idx)*0.1));
idx_train=idx;
idx_train(idx_validation)=[];

n_class=length(unique(label_train));

if(iscell(feature_train))
    input=[];
    for i=1:length(feature_train)
        input_tmp=reshape(feature_train{1,i},[size(feature_train{1,i},1),Q+1,1,size(feature_train{1,i},2)/(Q+1)]);
        input_tmp=permute(input_tmp,[4,2,3,1]);
        input=cat(1,input,input_tmp);
    end
else
    input=reshape(feature_train,[size(feature_train,1),Q+1,1,size(feature_train,2)/(Q+1)]);
    input=permute(input,[4,2,3,1]);
end

input_validation=input(:,:,:,idx_validation);
label_valitation=label_train(idx_validation);
input(:,:,:,idx_validation)=[];
label_train(idx_validation)=[];

% epoch_num=max(3,min(30,round(10000/(length(label_train)/miniBatchSize))));
epoch_num=min(30,round(10000/(length(label_train)/miniBatchSize)));

% idx_shuffle=randperm(size(input,4));
% input=input(:,:,:,idx_shuffle);
% label_train=label_train(idx_shuffle);
dim1=size(input,1);
dim2=size(input,2);

% kernel1=3;
% kernel2=min(3,Q+1);
% Stride1=2;
% Stride2=min(floor(power(dim2,1/3)),2);

kernel1=max(ceil(power(dim1,1/3)),1);
kernel2=max(ceil(power(dim2,1/3)),1);
Stride1=max(ceil(power(dim1,1/3)/2),1);
Stride2=max(ceil(power(dim2,1/3)/2),1);

layers = [imageInputLayer([dim1 dim2 1],'Normalization','none')
          convolution2dLayer([kernel1,kernel2],32,'Stride',1,'Padding','same')%8
          batchNormalizationLayer
          reluLayer
          maxPooling2dLayer([Stride1,Stride2],'Stride',[Stride1,Stride2],'Padding','same')
          dropoutLayer(0.3)

          convolution2dLayer([kernel1,kernel2],64,'Stride',1,'Padding','same')
          batchNormalizationLayer
          reluLayer
          maxPooling2dLayer([Stride1,Stride2],'Stride',[Stride1,Stride2],'Padding','same')
          dropoutLayer(0.3)
          

          

%           convolution2dLayer([kernel1,kernel2],64,'Stride',1,'Padding','same')
%           batchNormalizationLayer
%           reluLayer
%           maxPooling2dLayer([Stride1,Stride2],'Stride',1,'Padding','same')
%           dropoutLayer(0.3)
% 
%           convolution2dLayer([kernel1,kernel2],64,'Stride',1,'Padding','same')
%           batchNormalizationLayer
%           reluLayer
%           maxPooling2dLayer([Stride1,Stride2],'Stride',1,'Padding','same')
%           dropoutLayer(0.3)
% 
%           convolution2dLayer([kernel1,kernel2],64,'Stride',1,'Padding','same')
%           batchNormalizationLayer
%           reluLayer
%           maxPooling2dLayer([Stride1,Stride2],'Stride',1,'Padding','same')
%           dropoutLayer(0.3)





          convolution2dLayer([kernel1,kernel2],128,'Stride',1,'Padding','same')
          batchNormalizationLayer
          reluLayer
          maxPooling2dLayer([Stride1,Stride2],'Stride',[Stride1,Stride2],'Padding','same')
          dropoutLayer(0.3)

          fullyConnectedLayer(200)
          reluLayer
          fullyConnectedLayer(100)
          reluLayer
          fullyConnectedLayer(20)
          reluLayer
          fullyConnectedLayer(n_class)
          softmaxLayer
          classificationLayer];


% options = trainingOptions('sgdm', ...
%     'MiniBatchSize',miniBatchSize, ...
%     'MaxEpochs',epoch_num, ...
%     'InitialLearnRate',1e-3, ...
%     'LearnRateSchedule','piecewise', ...
%     'LearnRateDropFactor',0.1, ...
%     'LearnRateDropPeriod',50, ...
%     'Plots','training-progress', ...
%     'Shuffle','every-epoch', ...
%     'Verbose',false);
options = trainingOptions('sgdm', ...
    'MiniBatchSize',miniBatchSize, ...
    'MaxEpochs',epoch_num, ...
    'ValidationData',{input_validation,categorical(label_valitation)}, ...
    'ValidationFrequency',100, ...
    'OutputNetwork','best-validation-loss', ...
    'InitialLearnRate',1e-3, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.1, ...
    'LearnRateDropPeriod',50, ...
    'Shuffle','every-epoch', ...
    'Verbose',false);
net = trainNetwork(input,categorical(label_train),layers,options);





% epoch_num=max(5,min(25,round(3000/(length(label_train)/miniBatchSize))));
% 
% n_class=length(unique(label_train));
% 
% if(iscell(feature_train))
%     input=[];
%     for i=1:length(feature_train)
%         input_tmp=reshape(feature_train{1,i},[size(feature_train{1,i},1),Q+1,1,size(feature_train{1,i},2)/(Q+1)]);
%         input_tmp=permute(input_tmp,[4,2,3,1]);
%         input=cat(1,input,input_tmp);
%     end
% else
%     input=reshape(feature_train,[size(feature_train,1),Q+1,1,size(feature_train,2)/(Q+1)]);
%     input=permute(input,[4,2,3,1]);
% end
% 
% % idx_shuffle=randperm(size(input,4));
% % input=input(:,:,:,idx_shuffle);
% % label_train=label_train(idx_shuffle);
% dim1=size(input,1);
% dim2=size(input,2);
% 
% kernel1=3;
% kernel2=min(3,Q+1);
% poolStride1=2;
% poolStride2=min(floor(power(Q+1,1/3)),2);
% 
% layers = [imageInputLayer([dim1 dim2 1],'Normalization','none')
%           convolution2dLayer([kernel1,kernel2],8,'Stride',1,'Padding','same')
%           batchNormalizationLayer
%           reluLayer
%           maxPooling2dLayer([poolStride1,poolStride2],'Stride',[poolStride1,poolStride2])
%           dropoutLayer(0.3)
% 
%           convolution2dLayer([kernel1,kernel2],16,'Stride',1,'Padding','same')
%           batchNormalizationLayer
%           reluLayer
%           maxPooling2dLayer([poolStride1,poolStride2],'Stride',[poolStride1,poolStride2])
%           dropoutLayer(0.3)
% 
%           convolution2dLayer([kernel1,kernel2],32,'Stride',1,'Padding','same')
%           batchNormalizationLayer
%           reluLayer
%           maxPooling2dLayer([poolStride1,poolStride2],'Stride',[poolStride1,poolStride2])
%           dropoutLayer(0.3)
% 
%           fullyConnectedLayer(200)
%           reluLayer
%           fullyConnectedLayer(100)
%           reluLayer
%           fullyConnectedLayer(20)
%           reluLayer
%           fullyConnectedLayer(n_class)
%           softmaxLayer
%           classificationLayer];
% 
% % validationFrequency = floor(length(label)/miniBatchSize);
% % options = trainingOptions('sgdm', ...
% %     'MiniBatchSize',miniBatchSize, ...
% %     'MaxEpochs',epoch_num, ...
% %     'InitialLearnRate',1e-3, ...
% %     'LearnRateSchedule','piecewise', ...
% %     'LearnRateDropFactor',0.1, ...
% %     'LearnRateDropPeriod',50, ...
% %     'Plots','training-progress', ...
% %     'Shuffle','every-epoch', ...
% %     'Verbose',false);
% options = trainingOptions('sgdm', ...
%     'MiniBatchSize',miniBatchSize, ...
%     'MaxEpochs',epoch_num, ...
%     'InitialLearnRate',1e-3, ...
%     'LearnRateSchedule','piecewise', ...
%     'LearnRateDropFactor',0.1, ...
%     'LearnRateDropPeriod',50, ...
%     'Shuffle','every-epoch', ...
%     'Verbose',false);
% net = trainNetwork(input,categorical(label_train),layers,options);