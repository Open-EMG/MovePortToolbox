function label_out=label_window(label,window_length,window_step,fs)

window_length_sample=round(window_length/1000*fs);
window_step_sample=round(window_step/1000*fs);
len=length(label);

idx_window=0;
for idx_sample=1:window_step_sample:len-window_length_sample+1
    idx_window=idx_window+1;
    label_seg=label(idx_sample:idx_sample+window_length_sample-1);
    label_out(idx_window,1)=mode(label_seg);
end