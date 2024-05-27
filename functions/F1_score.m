function F1=F1_score(label_predict,label_test,tolerance)


event_truth=cell(2,1);
event_predict=cell(2,1);

    
for m=2:length(label_test)
    if(label_test(m-1)==0 && label_test(m)==1)
        event_truth{1}=[event_truth{1},m];
    end
    if(label_test(m-1)==1 && label_test(m)==0)
        event_truth{2}=[event_truth{2},m];
    end
end
    
for m=2:length(label_predict)
    if(label_predict(m-1)==0 && label_predict(m)==1)
        event_predict{1}=[event_predict{1},m];
    end
    if(label_predict(m-1)==1 && label_predict(m)==0)
        event_predict{2}=[event_predict{2},m];
    end
end

for m=1:2
    event_truth_tmp=event_truth{m};
    event_predict_tmp=event_predict{m};
    TP=0;
    FP=0;
    FP_not_match=0;
    FP_additional_match=0;
    FN=0;
    for k=1:length(event_truth_tmp)
        for u=1:length(event_predict_tmp)
            if(abs(event_predict_tmp(u)-event_truth_tmp(k))<=tolerance)
                TP=TP+1;
                break;
            end
        end
    end
    for u=1:length(event_predict_tmp)
        count_flag=1;
        for k=1:length(event_truth_tmp)
            if(abs(event_predict_tmp(u)-event_truth_tmp(k))<=tolerance)
                count_flag=0;
                break;
            end
        end
        FP_not_match=FP_not_match+count_flag;
    end

    for k=1:length(event_truth_tmp)
        count_match=0;
        for u=1:length(event_predict_tmp)
            if(abs(event_predict_tmp(u)-event_truth_tmp(k))<=tolerance)
                count_match=count_match+1;
            end
        end
        if(count_match==0)
            FN=FN+1;
        else
            FP_additional_match=FP_additional_match+(count_match-1);
        end
    end
    FP=FP_not_match+FP_additional_match;
    P=TP/(TP+FP);
    R=TP/(TP+FN);
    F1(m)=2*P*R/(P+R);
end

    
    