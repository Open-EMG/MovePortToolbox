function label_2 = label_2_stages(force, fs, min_duration_gap, var_names)
    % Define filter parameters
    order = 1;
    cutoff = 0.1;
    [b, a] = butter(order, cutoff, 'low');
    
    % Apply the filter to the force data
    force_filter = filtfilt(b, a, force(find(var_names=='R_force'), :) - min(force(find(var_names=='R_force'), :)));
    
    % Initialize the label_2 array
    label_2 = zeros(length(force_filter),1);
    
    force_sort=sort(force_filter);
    threshold=force_sort(round(length(force_sort)*0.03))+50;
    % Set labels based on filter results
    for i = 1:length(force_filter)
        if force_filter(i) < threshold %50
            label_2(i,1) = 1;
        else
            label_2(i,1) = 0;
        end
    end

    label_2=label_filter(label_2,fs, min_duration_gap);



end