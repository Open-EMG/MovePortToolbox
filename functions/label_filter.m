function label_smooth=label_filter(label,fs,min_duration_gap)

label_smooth=label;

idx_up_all=find([false;diff(label)==1]);
idx_down_all=find([false;diff(label)==-1]);

if(length(idx_up_all)>length(idx_down_all))
    idx_down_all=[idx_down_all;length(label)];
end
if(length(idx_down_all)>length(idx_up_all))
    idx_up_all=[idx_up_all;length(label)];
end

if(label_smooth(1)==0)    
    for k=1:length(idx_up_all)
        if((idx_down_all(k)-idx_up_all(k))<(min_duration_gap/1000*fs))
            label_smooth(idx_up_all(k):idx_down_all(k))=0;
        end
    end
    for k=1:length(idx_up_all)-1
        if((idx_up_all(k+1)-idx_down_all(k))<(min_duration_gap/1000*fs))
            label_smooth(idx_down_all(k):idx_up_all(k+1))=1;
        end
    end
end

if(label_smooth(1)==1)    
    for k=1:length(idx_up_all)
        if((idx_up_all(k)-idx_down_all(k))<(min_duration_gap/1000*fs))
            label_smooth(idx_down_all(k):idx_up_all(k))=1;
        end
    end
    for k=1:length(idx_down_all)-1
        if((idx_down_all(k+1)-idx_up_all(k))<(min_duration_gap/1000*fs))
            label_smooth(idx_up_all(k):idx_down_all(k+1))=0;
        end
    end
end
