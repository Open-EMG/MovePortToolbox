function cycle_duration=period_evaluate(label)

idx_up=find([false;diff(label)==1]);

cycle_duration_all=diff(idx_up);

cycle_duration=median(cycle_duration_all);