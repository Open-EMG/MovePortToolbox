clear all;
close all;

path_data=['../data/25/ground_gait/mocap_1.csv'];

fs_mocap=100;

mocap=readtable(path_data);
mocap=mocap(2:end,:);

skeleton_construct_plot(mocap,fs_mocap);
