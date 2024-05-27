clear all;
close all;

path='..';

min_duration_gap=200;

fs=100; % resampling rate

forward_direction=[1,0,0];

fs_ips=60;
fs_cop=60;
fs_mocap=100;


label_2=[];
label_subject=[];
label_mode=[];

idx_cycle=0;

N_subject=25

for i=1:N_subject % subjects 1 to 25
    i
    if(i<=24)
        mode_types=dir([path,'/data/',num2str(i)]);
        mode_types=mode_types(contains({mode_types.name},'treadmill'));
    else
        mode_types=dir([path,'/data/',num2str(i)]);
        mode_types=mode_types(contains({mode_types.name},'ground_gait'));
    end

    for idx_mode_type=1:length(mode_types)
        mode_type=mode_types(idx_mode_type,1).name;
        if(contains(mode_type,'normal'))
            mode=1;
        elseif(contains(mode_type,'dragging'))
            mode=2;
        elseif(contains(mode_type,'leghigh'))
            mode=3;
        else
            mode=4;
        end

        folder=[path,'/data/',num2str(i),'/',mode_type];
        files=dir(folder);
        files = files(contains({files.name},'mocap'));
    
        data_ips=[];
        data_cop=[];
        data_mocap=[];
    
    
        segment_name={};
        for k=1:length(files)
            filename=files(k).name;
            idx_str=strfind(filename,'_');
            segment_name{k,1}=filename(idx_str(1)+1:end);
        end
        segment_name = segment_name(~cellfun(@isempty, segment_name));
    
        for k=1:length(segment_name)
            mocap=readtable([folder,'/mocap_',segment_name{k,:}]);
            channels=mocap(2:end,1);
            mocap=readmatrix([folder,'/mocap_',segment_name{k,:}]);
            mocap=mocap(2:end,2:end);
   
            cop=readmatrix([folder,'/cop_',segment_name{k,:}],'OutputType', 'string');
            var_names=cop(2:end,1);
            cop=readmatrix([folder,'/cop_',segment_name{k,:}]);
            cop=cop(2:end,2:end);
            cop = resample(cop',size(mocap,2),size(cop,2))';

            label_gait=label_2_stages(cop,fs,min_duration_gap,var_names);

            %segment gait cycle
            idx_gait_onset_all=(find(diff(label_gait)==1));
            for m=1:length(idx_gait_onset_all)-1
                idx_cycle=idx_cycle+1;
                data_mocap=mocap(:,idx_gait_onset_all(m)+1:idx_gait_onset_all(m+1));
                data_mocap = resample(data_mocap',51,size(data_mocap,2))';
                feature_mocap_sample=feature_extraction_kinematic(data_mocap,channels,1000/fs,1000/fs,forward_direction,fs);  
                feature_mocap(idx_cycle,:)=reshape(feature_mocap_sample,[1,size(feature_mocap_sample,1)*size(feature_mocap_sample,2)]);
                label_subject(idx_cycle)=i;
                label_mode(idx_cycle)=mode;
            end
        end
    end
end

for i=1:N_subject 
    idx_cycle=find((label_mode==1)&(label_subject~=i));
    feature_modelling=feature_mocap(idx_cycle,:);
    [U,S,~] = svd(feature_modelling','econ'); 
    [~,dim]=min(abs(cumsum(sum(S))/sum(sum(S))-0.98));
    C_modelling=U(:,1:dim)'*feature_modelling';
    C_template=mean(C_modelling,2);
    for mode=1:4
        idx_cycle=find((label_mode==mode)&(label_subject==i));
        if(length(idx_cycle)==0)
            distance(i,mode)=nan;
        else
            feature_evaluate=feature_mocap(idx_cycle,:);
            C_evaluate=U(:,1:dim)'*feature_evaluate';
            distance(i,mode)=sqrt(mean((mean(C_evaluate,2)-C_template).^2));
        end
    end
end

for i=1:N_subject 
    idx_subject=[1:N_subject];
    idx_subject(i)=[];
    GDI_raw_modelling=log(distance(idx_subject,:));
    GDI_raw_test=log(distance(i,:));

    GDI_mean=mean(GDI_raw_modelling(:,1),'omitnan');
    GDI_std=std(GDI_raw_modelling(:,1),'omitnan');
    
    zGDI=(GDI_raw_test-GDI_mean)/GDI_std;
    GDI(i,:)=100-10*zGDI;
end
mean(GDI,'omitnan');
p1=signrank(GDI(1:16,1),GDI(1:16,2))
p1=signrank(GDI(1:16,1),GDI(1:16,3))
