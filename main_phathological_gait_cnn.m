clear all;
close all;

path='..'; % path of the MovePort dataset on your device

mode_type='ground_gait';

miniBatchSize=16;

min_duration_gap=200;

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

train=0.7;

id_subject=25; % subject 25 is a spastic paraplegia patient

feature_emg=[];
feature_ips=[];
feature_cop=[];
feature_imu=[];
feature_mocap=[];

label_2=[];

folder=[path,'/data/',num2str(id_subject),'/',mode_type];
files=dir(folder);
files = files(~endsWith({files.name},{'.avi'}));

data_ips=[];
data_cop=[];

segment_name={};
for k=3:length(files)
    filename=files(k).name;
    idx_str1=strfind(filename,'_');
    idx_str2=strfind(filename,'.');
    segment_name{k-2,1}=filename(idx_str1+1:idx_str2-1);
end
segment_name = segment_name(~cellfun(@isempty, segment_name));
segment_name=unique(str2double(segment_name),'rows');

for k=1:size(segment_name,1)
    emg=readmatrix([folder,'/emg_',num2str(segment_name(k,:)),'.csv']);
    emg=emg(2:end,2:end);
    feature_emg_sample=feature_extraction_emg(emg,window_length,window_step,fs);
    feature_emg=[feature_emg,{feature_emg_sample}];

    ips=readmatrix([folder,'/ips_',num2str(segment_name(k,:)),'.csv']);
    ips=ips(2:end,2:end);
    ips = resample(ips',size(emg,2),size(ips,2))';
    ips2d=reshape(ips,[11,size(ips,1)/11/2,2,size(ips,2)]);
    data_ips=[data_ips,{ips}];
    feature_ips_sample=feature_extraction_mean(ips,window_length,window_step,fs);
    feature_ips=[feature_ips,{feature_ips_sample}];

    cop=readmatrix([folder,'/cop_',num2str(segment_name(k,:)),'.csv'],'OutputType', 'string');
    var_names=cop(2:end,1);
    cop=readmatrix([folder,'/cop_',num2str(segment_name(k,:)),'.csv']);
    cop=cop(2:end,2:end);
    cop = resample(cop',size(emg,2),size(cop,2))';
    data_cop=[data_cop,{cop}];
    feature_cop_sample=feature_extraction_mean(cop,window_length,window_step,fs);
    feature_cop=[feature_cop,{feature_cop_sample}];

    mocap=readmatrix([folder,'/mocap_',num2str(segment_name(k,:)),'.csv']);
    mocap=mocap(2:end,2:end);
    mocap = resample(mocap',size(emg,2),size(mocap,2))';
    feature_mocap_sample=feature_extraction_diff(mocap,window_length,window_step,fs);
    feature_mocap=[feature_mocap,{feature_mocap_sample}];

    imu=readmatrix([folder,'/imu_',num2str(segment_name(k,:)),'.csv']);
    imu=imu(2:end,2:end);
    imu = resample(imu',size(emg,2),size(imu,2))';
    feature_imu_sample=feature_extraction_diff(imu,window_length,window_step,fs);
    feature_imu=[feature_imu,{feature_imu_sample}];
end



    

for m=1:length(data_cop)
    label_2_append=label_window(label_2_stages(data_cop{1,m},fs,min_duration_gap,var_names),window_length,window_step,fs);
    label_2 = [label_2,{label_2_append}];
end



[feature_emg_timelag,label_2_timelag,label_trial]=data_prepare(feature_emg,label_2,Q);
[feature_imu_timelag,~,~]=data_prepare(feature_imu,label_2,Q);
[feature_mocap_timelag,~,~]=data_prepare(feature_mocap,label_2,Q);


N=length(label_2_timelag);
idx_all=[1:N];
idx_train=idx_all(1:round(N*train));
idx_test=idx_all(round(N*train)+Q+1:N);
label_2_train=label_2_timelag(idx_train);
label_2_test=label_2_timelag(idx_test);
label_trial_test=label_trial(idx_test);

cycle_duration=period_evaluate(label_2_timelag);
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
f1_emg_2=F1_score(label_predict_emg_2,label_2_test,tolerance)

label_predict_mocap_2=gait_state_logic(cnn_predict(mdl_mocap_2,feature_test_mocap,Q),label_trial_test,min_duration_gap/window_step);
f1_mocap_2=F1_score(label_predict_mocap_2,label_2_test,tolerance)

label_predict_imu_2=gait_state_logic(cnn_predict(mdl_imu_2,feature_test_imu,Q),label_trial_test,min_duration_gap/window_step);
f1_imu_2=F1_score(label_predict_imu_2,label_2_test,tolerance)

label_predict_emg_mocap_2=gait_state_logic(cnn_predict(mdl_emg_mocap_2,{feature_test_emg,feature_test_mocap},Q),label_trial_test,min_duration_gap/window_step);
f1_emg_mocap_2=F1_score(label_predict_emg_mocap_2,label_2_test,tolerance)

label_predict_emg_imu_2=gait_state_logic(cnn_predict(mdl_emg_imu_2,{feature_test_emg,feature_test_imu},Q),label_trial_test,min_duration_gap/window_step);
f1_emg_imu_2=F1_score(label_predict_emg_imu_2,label_2_test,tolerance)

label_predict_mocap_imu_2=gait_state_logic(cnn_predict(mdl_mocap_imu_2,{feature_test_mocap,feature_test_imu},Q),label_trial_test,min_duration_gap/window_step);
f1_mocap_imu_2=F1_score(label_predict_mocap_imu_2,label_2_test,tolerance)

label_predict_emg_mocap_imu_2=gait_state_logic(cnn_predict(mdl_emg_mocap_imu_2,{feature_test_emg,feature_test_mocap,feature_test_imu},Q),label_trial_test,min_duration_gap/window_step);
f1_emg_mocap_imu_2=F1_score(label_predict_emg_mocap_imu_2,label_2_test,tolerance)



% save([path,'/results/f1_emg_2_',mode_type,'_cnn','_tolRate_',num2str(tolerance_rate*100),'_Q_',num2str(Q),'.mat'],'f1_emg_2');
% save([path,'/results/f1_imu_2_',mode_type,'_cnn','_tolRate_',num2str(tolerance_rate*100),'_Q_',num2str(Q),'.mat'],'f1_imu_2');
% save([path,'/results/f1_mocap_2_',mode_type,'_cnn','_tolRate_',num2str(tolerance_rate*100),'_Q_',num2str(Q),'.mat'],'f1_mocap_2');
% save([path,'/results/f1_emg_mocap_2_',mode_type,'_cnn','_tolRate_',num2str(tolerance_rate*100),'_Q_',num2str(Q),'.mat'],'f1_emg_mocap_2');
% save([path,'/results/f1_emg_imu_2_',mode_type,'_cnn','_tolRate_',num2str(tolerance_rate*100),'_Q_',num2str(Q),'.mat'],'f1_emg_imu_2');
% save([path,'/results/f1_mocap_imu_2_',mode_type,'_cnn','_tolRate_',num2str(tolerance_rate*100),'_Q_',num2str(Q),'.mat'],'f1_mocap_imu_2');
% save([path,'/results/f1_emg_mocap_imu_2_',mode_type,'_cnn','_tolRate_',num2str(tolerance_rate*100),'_Q_',num2str(Q),'.mat'],'f1_emg_mocap_imu_2');



