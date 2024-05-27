function []=skeleton_construct_plot(mocap,fs)

channels=mocap{1:end,1};
sensors=mocap{1:3:end,1};
N_sensors=length(sensors);
for i=1:N_sensors
    sensors{i,1}=sensors{i,1}(1:end-2);
end

skeleton_graph=zeros(N_sensors,N_sensors);
index_fully_connect=[find(contains(sensors,'RA')),find(contains(sensors,'LA')),find(contains(sensors,'IJ')),find(contains(sensors,'C7')),find(contains(sensors,'T8'))];
skeleton_graph(index_fully_connect,index_fully_connect)=1;
index_fully_connect=[find(contains(sensors,'M_PSIS')),find(contains(sensors,'R_IAS')),find(contains(sensors,'L_IAS'))];
skeleton_graph(index_fully_connect,index_fully_connect)=1;
skeleton_graph(find(contains(sensors,'T8')),find(contains(sensors,'M_PSIS')))=1;
skeleton_graph(find(contains(sensors,'RA')),find(contains(sensors,'R_LHE')))=1;
skeleton_graph(find(contains(sensors,'R_LHE')),find(contains(sensors,'R_RS')))=1;
skeleton_graph(find(contains(sensors,'LA')),find(contains(sensors,'L_LHE')))=1;
skeleton_graph(find(contains(sensors,'L_LHE')),find(contains(sensors,'L_RS')))=1;
skeleton_graph(find(contains(sensors,'L_IAS')),find(contains(sensors,'L_FTC')))=1;
skeleton_graph(find(contains(sensors,'R_IAS')),find(contains(sensors,'R_FTC')))=1;
index_fully_connect=[find(contains(sensors,'L_FTC')),find(contains(sensors,'L_FME')),find(contains(sensors,'L_FLE'))];
skeleton_graph(index_fully_connect,index_fully_connect)=1;
index_fully_connect=[find(contains(sensors,'R_FTC')),find(contains(sensors,'R_FME')),find(contains(sensors,'R_FLE'))];
skeleton_graph(index_fully_connect,index_fully_connect)=1;
skeleton_graph(find(contains(sensors,'L_FME')),find(contains(sensors,'L_TTC')))=1;
skeleton_graph(find(contains(sensors,'L_FLE')),find(contains(sensors,'L_TTC')))=1;
skeleton_graph(find(contains(sensors,'R_FME')),find(contains(sensors,'R_TTC')))=1;
skeleton_graph(find(contains(sensors,'R_FLE')),find(contains(sensors,'R_TTC')))=1;
skeleton_graph(find(contains(sensors,'L_TTC')),find(contains(sensors,'L_CAL')))=1;
skeleton_graph(find(contains(sensors,'R_TTC')),find(contains(sensors,'R_CAL')))=1;
index_fully_connect=[find(contains(sensors,'L_MH1')),find(contains(sensors,'L_LM')),find(contains(sensors,'L_CAL'))];
skeleton_graph(index_fully_connect,index_fully_connect)=1;
index_fully_connect=[find(contains(sensors,'R_MH1')),find(contains(sensors,'R_LM')),find(contains(sensors,'R_CAL'))];
skeleton_graph(index_fully_connect,index_fully_connect)=1;

x_values=mocap(find(contains(channels,'_x')),:);
y_values=mocap(find(contains(channels,'_y')),:);
z_values=mocap(find(contains(channels,'_z')),:);
x_range=[min(min(x_values{:,2:end})),max(max(x_values{:,2:end}))];
y_range=[min(min(y_values{:,2:end})),max(max(y_values{:,2:end}))];
z_range=[min(min(z_values{:,2:end})),max(max(z_values{:,2:end}))];

for frame=1:round(fs/20):size(mocap,2)-1
    points=reshape(mocap{:,frame+1},[3,length(channels)/3])';
    scatter3(points(:,1),points(:,2),points(:,3));
    hold on
    for i=1:N_sensors
        for j=2:N_sensors
            if(skeleton_graph(i,j))
                point1=mocap{find(contains(channels,sensors{i,1})),frame+1};
                point2=mocap{find(contains(channels,sensors{j,1})),frame+1};
                plot3([point1(1),point2(1)],[point1(2),point2(2)],[point1(3),point2(3)]);
            end
        end
    end
    xlim(x_range);
    ylim(y_range);
    zlim(z_range);
    xlabel('x');
    ylabel('y');
    zlabel('z');
    hold off
    drawnow limitrate
end


