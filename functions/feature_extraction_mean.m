function feature=feature_extraction_mean(data,window_length,window_step,fs)

[dim1,dim2,dim3]=size(data);

window_length_sample=round(window_length/1000*fs);
window_step_sample=round(window_step/1000*fs);

feature=[];
for i=1:dim3
    feature_tmp=[];
    idx_window=0;
    for idx_sample=1:window_step_sample:dim2-window_length_sample+1
        idx_window=idx_window+1;
        feature_channel_window=[];
        for j=1:dim1
            data_seg=data(j,idx_sample:idx_sample+window_length_sample-1,i);
            feature_channel_window=[feature_channel_window;mean(data_seg)];
        end
        feature_channel_window=reshape(feature_channel_window,[numel(feature_channel_window),1]);
        feature_tmp(idx_window,:)=feature_channel_window;
    end
    feature(:,:,i)=feature_tmp;
end
