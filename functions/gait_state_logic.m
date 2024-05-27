function label_smooth=gait_state_logic(label_predict,label_trial_test,min_window_gap)

min_duration=100; %100 ms

label_trial_unique=unique(label_trial_test);

for k=1:length(label_trial_unique)
    idx=find(label_trial_test==label_trial_unique(k));
    label_predict_tmp=label_predict(idx);
%     label_smooth_tmp=label_predict_tmp;
    idx_up=0;
    idx_down=0;
    for m=2:length(label_predict_tmp)
        if( (label_predict_tmp(m-1)==0) && (label_predict_tmp(m)==1) )
            if((m-idx_down)<=min_window_gap)
                label_predict_tmp(m)=0;
            else
                idx_up=m;
                idx_down=0;
            end
        end
        if( (label_predict_tmp(m-1)==1) && (label_predict_tmp(m)==0) )
            if((m-idx_up)<=min_window_gap)
                label_predict_tmp(m)=1;
            else
                idx_up=0;
                idx_down=m;
            end
        end
    end
    label_smooth(idx,:)=label_predict_tmp;
end


% label_trial_unique=unique(label_trial_test);
% 
% n_class=length(unique(label_predict));
% 
% if(n_class==4)
%     for k=1:length(label_trial_unique)
%         idx=find(label_trial_test==label_trial_unique(k));
%         label_predict_tmp=label_predict(idx);
%     %     label_smooth_tmp=label_predict_tmp;
%         for m=2:length(label_predict_tmp)
%             if( (label_predict_tmp(m-1)==0) && ( (label_predict_tmp(m)==2) || (label_predict_tmp(m)==3) ) )
%                 label_predict_tmp(m)=0;
%             end
%             if( (label_predict_tmp(m-1)==1) && ( (label_predict_tmp(m)==0) || (label_predict_tmp(m)==3) ) )
%                 label_predict_tmp(m)=1;
%             end
%             if( (label_predict_tmp(m-1)==2) && ( (label_predict_tmp(m)==0) || (label_predict_tmp(m)==1) ) )
%                 label_predict_tmp(m)=2;
%             end
%             if( (label_predict_tmp(m-1)==3) && ( (label_predict_tmp(m)==1) || (label_predict_tmp(m)==2) ) )
%                 label_predict_tmp(m)=3;
%             end
%         end
%         label_smooth(idx,:)=label_predict_tmp;
%     end
% else
%     label_smooth=label_predict;
% end
