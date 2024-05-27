function mocap_transform=mocap_ref(mocap,sensors,sensor_ref)

for i=1:size(sensors,1)
    sensor=sensors{i,1}{1,1};
    if(strcmp(sensor,[sensor_ref,'_x']))
        sensor_ref_id=i;
        break;
    end
end

mocap_transform=mocap;
for i=1:3:size(mocap_transform,1)
    mocap_transform(i,:)=mocap_transform(i,:)-mocap_transform(sensor_ref_id,:);
    mocap_transform(i+1,:)=mocap_transform(i+1,:)-mocap_transform(sensor_ref_id+1,:);
    mocap_transform(i+2,:)=mocap_transform(i+2,:)-mocap_transform(sensor_ref_id+2,:);
end
mocap_transform(sensor_ref_id:sensor_ref_id+2,:)=[];
    