clear all;
close all;

path='..'; % path of the MovePort dataset on your device

postures={'still','back','forward','halfsquat'};
n_class=length(postures);

miniBatchSize=16;

Q=0;

window_length=50; % window length: 50 ms
window_step=50; % window sliding step: 50 ms

fs=2000; % sampling rate (after resampling)

fs_emg=2000;
fs_imu=100;
fs_ips=60;
fs_cop=60;
fs_mocap=100;


feature_emg=[];
feature_ips=[];
feature_imu=[];
feature_mocap=[];
label_all=[];
label_subject_all=[];
for i=1:24 % subjects 1 to 24
    i
    for j=1:length(postures)
        folder=[path,'/data/',num2str(i),'/',postures{1,j}];
        files=dir(folder);
        files = files(~endsWith({files.name},{'.avi'}));

        segment_name={};
        for k=3:length(files)
            filename=files(k).name;
            idx_str=strfind(filename,'_');
            segment_name{k-2,1}=filename(idx_str+1:end);
        end
        segment_name = segment_name(~cellfun(@isempty, segment_name));
        segment_name=unique(cell2mat(segment_name),'rows');

        for k=1:size(segment_name,1)
            emg=readmatrix([folder,'/emg_',segment_name(k,:)]);
            emg=emg(2:end,2:end);
            feature_emg_sample=feature_extraction_emg(emg,window_length,window_step,fs);
            feature_emg=[feature_emg,{feature_emg_sample}];
            label_sample=(j-1)*ones(size(feature_emg_sample,1),1);
            label_all=[label_all,{label_sample}];
            label_subject_all=[label_subject_all,{i*ones(size(feature_emg_sample,1),1)}];
    
            ips=readmatrix([folder,'/ips_',segment_name(k,:)]);
            ips=ips(2:end,2:end);
            ips = resample(ips',size(emg,2),size(ips,2))';
            ips2d=reshape(ips,[11,size(ips,1)/11/2,2,size(ips,2)]);
            feature_ips_sample=feature_extraction_mean(ips,window_length,window_step,fs);
            feature_ips=[feature_ips,{feature_ips_sample}];
    
            mocap=readtable([folder,'/mocap_',segment_name(k,:)]);
            sensors=mocap(2:end,1);
            sensor_ref='IJ';
            mocap=readmatrix([folder,'/mocap_',segment_name(k,:)]);
            mocap=mocap(2:end,2:end);
            mocap = resample(mocap',size(emg,2),size(mocap,2))';
            feature_mocap_sample=feature_extraction_diff(mocap,window_length,window_step,fs);
            mocap_transform=mocap_ref(mocap,sensors,sensor_ref);
            feature_mocap_sample2=feature_extraction_mean(mocap_transform,window_length,window_step,fs);     
            feature_mocap=[feature_mocap,{[feature_mocap_sample,feature_mocap_sample2]}];
    
            imu=readmatrix([folder,'/imu_',segment_name(k,:)]);
            imu=imu(2:end,2:end);
            imu = resample(imu',size(emg,2),size(imu,2))';
            feature_imu_sample=feature_extraction_diff(imu,window_length,window_step,fs);
            feature_imu=[feature_imu,{feature_imu_sample}];
        end
    end
end

[~,label_subject,~]=data_prepare(feature_emg,label_subject_all,Q);
[feature_emg,label,~]=data_prepare(feature_emg,label_all,Q);
[feature_ips,~,~]=data_prepare(feature_ips,label_all,Q);
[feature_imu,~,~]=data_prepare(feature_imu,label_all,Q);
[feature_mocap,~,~]=data_prepare(feature_mocap,label_all,Q);

for i=1:24


    N=length(label_subject);
    idx_test=find(label_subject==i);
    idx_train=[1:N];
    idx_train(idx_test)=[];
    
    label_train=label(idx_train);
    label_test=label(idx_test);

    feature_train_emg=feature_emg(idx_train,:);
    feature_test_emg=feature_emg(idx_test,:);
    feature_train_ips=feature_ips(idx_train,:);
    feature_test_ips=feature_ips(idx_test,:);
    feature_train_imu=feature_imu(idx_train,:);
    feature_test_imu=feature_imu(idx_test,:);
    feature_train_mocap=feature_mocap(idx_train,:);
    feature_test_mocap=feature_mocap(idx_test,:);

    idx=find(std(feature_train_emg)<1e-10);
    feature_train_emg(:,idx)=[];
    feature_test_emg(:,idx)=[];
    idx=find(std(feature_train_ips)<1e-10);
    feature_train_ips(:,idx)=[];
    feature_test_ips(:,idx)=[];
    idx=find(std(feature_train_imu)<1e-10);
    feature_train_imu(:,idx)=[];
    feature_test_imu(:,idx)=[];
    idx=find(std(feature_train_mocap)<1e-10);
    feature_train_mocap(:,idx)=[];
    feature_test_mocap(:,idx)=[];

    idx=find(std(feature_test_emg)<1e-10);
    feature_train_emg(:,idx)=[];
    feature_test_emg(:,idx)=[];
    idx=find(std(feature_test_ips)<1e-10);
    feature_train_ips(:,idx)=[];
    feature_test_ips(:,idx)=[];
    idx=find(std(feature_test_imu)<1e-10);
    feature_train_imu(:,idx)=[];
    feature_test_imu(:,idx)=[];
    idx=find(std(feature_test_mocap)<1e-10);
    feature_train_mocap(:,idx)=[];
    feature_test_mocap(:,idx)=[];

    [feature_train_emg,feature_test_emg]=feature_normalize(feature_train_emg,feature_test_emg);
    [feature_train_ips,feature_test_ips]=feature_normalize(feature_train_ips,feature_test_ips);
    [feature_train_imu,feature_test_imu]=feature_normalize(feature_train_imu,feature_test_imu);
    [feature_train_mocap,feature_test_mocap]=feature_normalize(feature_train_mocap,feature_test_mocap);

    mdl_emg = cnn_fit(feature_train_emg,label_train,Q,miniBatchSize);
    mdl_ips = cnn_fit(feature_train_ips,label_train,Q,miniBatchSize);
    mdl_mocap = cnn_fit(feature_train_mocap,label_train,Q,miniBatchSize);
    mdl_imu = cnn_fit(feature_train_imu,label_train,Q,miniBatchSize);
    mdl_emg_mocap = cnn_fit({feature_train_emg,feature_train_mocap},label_train,Q,miniBatchSize);
    mdl_emg_imu = cnn_fit({feature_train_emg,feature_train_imu},label_train,Q,miniBatchSize);
    mdl_emg_ips = cnn_fit({feature_train_emg,feature_train_ips},label_train,Q,miniBatchSize);
    mdl_mocap_imu = cnn_fit({feature_train_mocap,feature_train_imu},label_train,Q,miniBatchSize);
    mdl_mocap_ips = cnn_fit({feature_train_mocap,feature_train_ips},label_train,Q,miniBatchSize);
    mdl_ips_imu = cnn_fit({feature_train_ips,feature_train_imu},label_train,Q,miniBatchSize);
    
    

    label_predict_emg=cnn_predict(mdl_emg,feature_test_emg,Q);
    acc_emg(i)=mean(label_predict_emg==label_test);
    mean(acc_emg)
    
    label_predict_ips=cnn_predict(mdl_ips,feature_test_ips,Q);
    acc_ips(i)=mean(label_predict_ips==label_test);
    mean(acc_ips)

    label_predict_imu=cnn_predict(mdl_imu,feature_test_imu,Q);
    acc_imu(i)=mean(label_predict_imu==label_test);
    mean(acc_imu)

    label_predict_mocap=cnn_predict(mdl_mocap,feature_test_mocap,Q);
    acc_mocap(i)=mean(label_predict_mocap==label_test);
    mean(acc_mocap)

    label_predict_emg_ips=cnn_predict(mdl_emg_ips,{feature_test_emg,feature_test_ips},Q);
    acc_emg_ips(i)=mean(label_predict_emg_ips==label_test);
    mean(acc_emg_ips)

    label_predict_emg_imu=cnn_predict(mdl_emg_imu,{feature_test_emg,feature_test_imu},Q);
    acc_emg_imu(i)=mean(label_predict_emg_imu==label_test);
    mean(acc_emg_imu)

    label_predict_emg_mocap=cnn_predict(mdl_emg_mocap,{feature_test_emg,feature_test_mocap},Q);
    acc_emg_mocap(i)=mean(label_predict_emg_mocap==label_test);
    mean(acc_emg_mocap)

    label_predict_mocap_ips=cnn_predict(mdl_mocap_ips,{feature_test_mocap,feature_test_ips},Q);
    acc_mocap_ips(i)=mean(label_predict_mocap_ips==label_test);
    mean(acc_mocap_ips)

    label_predict_mocap_imu=cnn_predict(mdl_mocap_imu,{feature_test_mocap,feature_test_imu},Q);
    acc_mocap_imu(i)=mean(label_predict_mocap_imu==label_test);
    mean(acc_mocap_imu)

    label_predict_ips_imu=cnn_predict(mdl_ips_imu,{feature_test_ips,feature_test_imu},Q);
    acc_ips_imu(i)=mean(label_predict_ips_imu==label_test);
    mean(acc_ips_imu)



%     save([path,'/results/accuracy_emg_posture_',num2str(n_class),'_class_','cnn_inter_subject.mat'],'acc_emg');
%     save([path,'/results/accuracy_ips_posture_',num2str(n_class),'_class_','cnn_inter_subject.mat'],'acc_ips');
%     save([path,'/results/accuracy_imu_posture_',num2str(n_class),'_class_','cnn_inter_subject.mat'],'acc_imu');
%     save([path,'/results/accuracy_mocap_posture_',num2str(n_class),'_class_','cnn_inter_subject.mat'],'acc_mocap');
%     save([path,'/results/accuracy_emg_ips_posture_',num2str(n_class),'_class_','cnn_inter_subject.mat'],'acc_emg_ips');
%     save([path,'/results/accuracy_emg_imu_posture_',num2str(n_class),'_class_','cnn_inter_subject.mat'],'acc_emg_imu');
%     save([path,'/results/accuracy_emg_mocap_posture_',num2str(n_class),'_class_','cnn_inter_subject.mat'],'acc_emg_mocap');
%     save([path,'/results/accuracy_mocap_ips_posture_',num2str(n_class),'_class_','cnn_inter_subject.mat'],'acc_mocap_ips');
%     save([path,'/results/accuracy_mocap_imu_posture_',num2str(n_class),'_class_','cnn_inter_subject.mat'],'acc_mocap_imu');
%     save([path,'/results/accuracy_ips_imu_posture_',num2str(n_class),'_class_','cnn_inter_subject.mat'],'acc_ips_imu');
end
