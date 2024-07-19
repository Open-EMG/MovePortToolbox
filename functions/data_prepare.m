function [data_out,label_out,label_trial]=data_prepare(data,label,Q)

if iscell(data)    == 0, data  = {data};    end  % Coerce to cell array.
if iscell(label)    == 0, data  = {label};    end  % Coerce to cell array.

for i=1:length(data)
    data_seg=data{1,i};
    label_seg=label{1,i};
    n_sample=size(data_seg,1);
    n_dim=size(data_seg,2);
    data_seg_timelag=zeros(n_sample-Q,n_dim*(Q+1));
    for t=Q+1:n_sample
        for q=1:Q+1
            data_seg_timelag(t-Q,(q-1)*n_dim+1:q*n_dim)=data_seg(q+t-(Q+1),:);
        end
    end
    data_out_cell{1,i}=data_seg_timelag;
    label_out_cell{1,i}=label{1,i}(Q+1:end);
end

data_out=[];
label_out=[];
label_trial=[];
for i=1:length(data_out_cell)
    data_out=[data_out;data_out_cell{1,i}];
    label_out=[label_out;label_out_cell{1,i}];
    label_trial=[label_trial;0*label_out_cell{1,i}+i];
end