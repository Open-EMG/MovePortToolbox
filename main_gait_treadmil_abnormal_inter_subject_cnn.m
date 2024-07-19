clear all;
close all;

path='..'; % path of the MovePort dataset on your device

mode_type='treadmill_dragging'; % select 'treadmill_dragging' or 'treadmill_leghigh'

f1_emg_2=[];
f1_mocap_2=[];
f1_imu_2=[];
f1_emg_mocap_2=[];
f1_emg_imu_2=[];
f1_mocap_imu_2=[];
f1_emg_mocap_imu_2=[];

min_duration_gap=200;

miniBatchSize=16;

window_length=50; % window length: 100 ms
window_step=50; % window sliding step: 50 ms

tolerance_rate=0.2;

Q=5;

fs=2000; % sampling rate (after resampling)

fs_emg=2000;
fs_imu=100;
fs_ips=60;
fs_cop=60;
fs_mocap=100;

feature_emg=[];
feature_ips=[];
feature_cop=[];
feature_imu=[];
feature_mocap=[];

label_2=[];
label_subject=[];

for i=1:24 % subjects 1 to 24
    i


    folder=[path,'/data/',num2str(i),'/',mode_type];
    files=dir(folder);
    files = files(~endsWith({files.name},{'.avi'}));

    data_ips=[];
    data_cop=[];

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

        ips=readmatrix([folder,'/ips_',segment_name(k,:)]);
        ips=ips(2:end,2:end);
        ips = resample(ips',size(emg,2),size(ips,2))';
        ips2d=reshape(ips,[11,size(ips,1)/11/2,2,size(ips,2)]);
        data_ips=[data_ips,{ips}];
        feature_ips_sample=feature_extraction_mean(ips,window_length,window_step,fs);
        feature_ips=[feature_ips,{feature_ips_sample}];

        cop=readmatrix([folder,'/cop_',segment_name(k,:)],'OutputType', 'string');
        var_names=cop(2:end,1);
        cop=readmatrix([folder,'/cop_',segment_name(k,:)]);
        cop=cop(2:end,2:end);
        cop = resample(cop',size(emg,2),size(cop,2))';
        data_cop=[data_cop,{cop}];
        feature_cop_sample=feature_extraction_mean(cop,window_length,window_step,fs);
        feature_cop=[feature_cop,{feature_cop_sample}];

        mocap=readmatrix([folder,'/mocap_',segment_name(k,:)]);
        mocap=mocap(2:end,2:end);
        mocap = resample(mocap',size(emg,2),size(mocap,2))';
        feature_mocap_sample=feature_extraction_diff(mocap,window_length,window_step,fs);
        feature_mocap=[feature_mocap,{feature_mocap_sample}];

        imu=readmatrix([folder,'/imu_',segment_name(k,:)]);
        imu=imu(2:end,2:end);
        imu = resample(imu',size(emg,2),size(imu,2))';
        feature_imu_sample=feature_extraction_diff(imu,window_length,window_step,fs);
        feature_imu=[feature_imu,{feature_imu_sample}];
    end

    for m=1:length(data_cop)
        label_2_append=label_window(label_2_stages(data_cop{1,m},fs, min_duration_gap, var_names),window_length,window_step,fs);
        label_2 = [label_2,{label_2_append}];
        label_subject=[label_subject,{i*ones(length(label_2_append),1)}];
    end
end

[feature_emg_timelag,label_2_timelag,label_trial]=data_prepare(feature_emg,label_2,Q);
[~,label_subject,~]=data_prepare(feature_emg,label_subject,Q);
[feature_imu_timelag,~,~]=data_prepare(feature_imu,label_2,Q);
[feature_mocap_timelag,~,~]=data_prepare(feature_mocap,label_2,Q);
        
    
for i=1:24
    i

    N=length(label_subject);
    idx_test=find(label_subject==i);
    idx_train=[1:N];
    idx_train(idx_test)=[];
    label_2_train=label_2_timelag(idx_train);
    label_2_test=label_2_timelag(idx_test);
    label_trial_test=label_trial(idx_test);

    if(length(label_2_test)==0)
        f1_emg_2(i,:)=nan;
        f1_mocap_2(i,:)=nan;
        f1_imu_2(i,:)=nan;
        f1_emg_mocap_2(i,:)=nan;
        f1_emg_imu_2(i,:)=nan;
        f1_mocap_imu_2(i,:)=nan;
        f1_emg_mocap_imu_2(i,:)=nan;
        continue;
    end

    cycle_duration=period_evaluate(label_2_test);
    tolerance=floor(tolerance_rate*cycle_duration);

    feature_train_emg=feature_emg_timelag(idx_train,:);
    feature_test_emg=feature_emg_timelag(idx_test,:);
    feature_train_mocap=feature_mocap_timelag(idx_train,:);
    feature_test_mocap=feature_mocap_timelag(idx_test,:);
    feature_train_imu=feature_imu_timelag(idx_train,:);
    feature_test_imu=feature_imu_timelag(idx_test,:);


    [feature_train_emg,feature_test_emg]=feature_normalize(feature_train_emg,feature_test_emg);
    [feature_train_mocap,feature_test_mocap]=feature_normalize(feature_train_mocap,feature_test_mocap);
    [feature_train_imu,feature_test_imu]=feature_normalize(feature_train_imu,feature_test_imu);

    mdl_emg_2 = cnn_fit(feature_train_emg,label_2_train,Q,miniBatchSize);
    mdl_mocap_2 = cnn_fit(feature_train_mocap,label_2_train,Q,miniBatchSize);
    mdl_imu_2 = cnn_fit(feature_train_imu,label_2_train,Q,miniBatchSize);
    mdl_emg_mocap_2 = cnn_fit({feature_train_emg,feature_train_mocap},label_2_train,Q,miniBatchSize);
    mdl_emg_imu_2 = cnn_fit({feature_train_emg,feature_train_imu},label_2_train,Q,miniBatchSize);
    mdl_mocap_imu_2 = cnn_fit({feature_train_mocap,feature_train_imu},label_2_train,Q,miniBatchSize);
    mdl_emg_mocap_imu_2 = cnn_fit({feature_train_emg,feature_train_mocap,feature_train_imu},label_2_train,Q,miniBatchSize);


    label_predict_emg_2=gait_state_logic(cnn_predict(mdl_emg_2,feature_test_emg,Q),label_trial_test,min_duration_gap/window_step);
    f1_emg_2(i,:)=F1_score(label_predict_emg_2,label_2_test,tolerance);
    mean(f1_emg_2)
    
    label_predict_mocap_2=gait_state_logic(cnn_predict(mdl_mocap_2,feature_test_mocap,Q),label_trial_test,min_duration_gap/window_step);
    f1_mocap_2(i,:)=F1_score(label_predict_mocap_2,label_2_test,tolerance);
    mean(f1_mocap_2)

    
    label_predict_imu_2=gait_state_logic(cnn_predict(mdl_imu_2,feature_test_imu,Q),label_trial_test,min_duration_gap/window_step);
    f1_imu_2(i,:)=F1_score(label_predict_imu_2,label_2_test,tolerance);
    mean(f1_imu_2)

    label_predict_emg_mocap_2=gait_state_logic(cnn_predict(mdl_emg_mocap_2,{feature_test_emg,feature_test_mocap},Q),label_trial_test,min_duration_gap/window_step);
    f1_emg_mocap_2(i,:)=F1_score(label_predict_emg_mocap_2,label_2_test,tolerance);
    mean(f1_emg_mocap_2)

    label_predict_emg_imu_2=gait_state_logic(cnn_predict(mdl_emg_imu_2,{feature_test_emg,feature_test_imu},Q),label_trial_test,min_duration_gap/window_step);
    f1_emg_imu_2(i,:)=F1_score(label_predict_emg_imu_2,label_2_test,tolerance);
    mean(f1_emg_imu_2)

    label_predict_mocap_imu_2=gait_state_logic(cnn_predict(mdl_mocap_imu_2,{feature_test_mocap,feature_test_imu},Q),label_trial_test,min_duration_gap/window_step);
    f1_mocap_imu_2(i,:)=F1_score(label_predict_mocap_imu_2,label_2_test,tolerance);
    mean(f1_mocap_imu_2)

    label_predict_emg_mocap_imu_2=gait_state_logic(cnn_predict(mdl_emg_mocap_imu_2,{feature_test_emg,feature_test_mocap,feature_test_imu},Q),label_trial_test,min_duration_gap/window_step);
    f1_emg_mocap_imu_2(i,:)=F1_score(label_predict_emg_mocap_imu_2,label_2_test,tolerance);
    mean(f1_emg_mocap_imu_2)


%     save([path,'/results/f1_emg_2_',mode_type,'_cnn','_tolRate_',num2str(tolerance_rate*100),'_Q_',num2str(Q),'_inter_subject.mat'],'f1_emg_2');
%     save([path,'/results/f1_imu_2_',mode_type,'_cnn','_tolRate_',num2str(tolerance_rate*100),'_Q_',num2str(Q),'_inter_subject.mat'],'f1_imu_2');
%     save([path,'/results/f1_mocap_2_',mode_type,'_cnn','_tolRate_',num2str(tolerance_rate*100),'_Q_',num2str(Q),'_inter_subject.mat'],'f1_mocap_2');
%     save([path,'/results/f1_emg_mocap_2_',mode_type,'_cnn','_tolRate_',num2str(tolerance_rate*100),'_Q_',num2str(Q),'_inter_subject.mat'],'f1_emg_mocap_2');
%     save([path,'/results/f1_emg_imu_2_',mode_type,'_cnn','_tolRate_',num2str(tolerance_rate*100),'_Q_',num2str(Q),'_inter_subject.mat'],'f1_emg_imu_2');
%     save([path,'/results/f1_mocap_imu_2_',mode_type,'_cnn','_tolRate_',num2str(tolerance_rate*100),'_Q_',num2str(Q),'_inter_subject.mat'],'f1_mocap_imu_2');
%     save([path,'/results/f1_emg_mocap_imu_2_',mode_type,'_cnn','_tolRate_',num2str(tolerance_rate*100),'_Q_',num2str(Q),'_inter_subject.mat'],'f1_emg_mocap_imu_2');

end

