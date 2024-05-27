function feature=feature_extraction_kinematic(data,channels,window_length,window_step,forward_direction,fs)

channels=channels{:,:};

[dim1,dim2,dim3]=size(data);

window_length_sample=round(window_length/1000*fs);
window_step_sample=round(window_step/1000*fs);


feature=[];
for i=1:dim3
    feature_tmp=[];
    idx_window=0;
    for idx_sample=1:window_step_sample:dim2-window_length_sample+1
        idx_window=idx_window+1;
        data_seg=data(:,idx_sample:idx_sample+window_length_sample-1,i);

        %calculate Pelvic tilt
        sagittal_laboratory_plane_v1=forward_direction;
        sagittal_laboratory_plane_v2=[0,0,1];
        find(contains(channels,'R_IAS'));
        IAS_mid_point=mean(data_seg(find(contains(channels,'R_IAS')),:)+data_seg(find(contains(channels,'L_IAS')),:),2)/2;
        sagittal_pelvic_axis=IAS_mid_point-mean(data_seg(find(contains(channels,'M_PSIS')),:),2);
        p_vector=project2plane(sagittal_pelvic_axis,sagittal_laboratory_plane_v1,sagittal_laboratory_plane_v2);
        pelvic_tilt=acos(dot(p_vector/norm(p_vector),forward_direction/norm(forward_direction)));
        if(p_vector(3)<0)
            pelvic_tilt=-pelvic_tilt; % positive value if front end is higher
        end

        %calculate Pelvic obliquity
        transverse_laboratory_axis=[1,-forward_direction(1)/forward_direction(2),0];
        transverse_laboratory_axis(find(isinf(transverse_laboratory_axis)))=100000000;
        transverse_laboratory_axis=transverse_laboratory_axis/norm(transverse_laboratory_axis);
        transverse_pelvic_axis=mean(data_seg(find(contains(channels,'R_IAS')),:)-data_seg(find(contains(channels,'L_IAS')),:),2);
        p_vector=project2plane(transverse_laboratory_axis,transverse_pelvic_axis,[0,0,1]);
        pelvic_obliquity=real(acos(dot(p_vector/norm(p_vector),transverse_pelvic_axis/norm(transverse_pelvic_axis))));
        if(pelvic_obliquity>pi/2)
            pelvic_obliquity=pi-pelvic_obliquity;
        end
        if(transverse_pelvic_axis(3)<0)
            pelvic_obliquity=-pelvic_obliquity; % positive value if right end is higher
        end

        %calculate Pelvic rotation
        transverse_pelvis_plane_v1=transverse_pelvic_axis;
        transverse_pelvis_plane_v2=sagittal_pelvic_axis;
        p_vector=project2plane(forward_direction,transverse_pelvis_plane_v1,transverse_pelvis_plane_v2);
        pelvic_rotation=acos(dot(p_vector/norm(p_vector),sagittal_pelvic_axis/norm(sagittal_pelvic_axis)));
        cross_product=cross(p_vector/norm(p_vector),sagittal_pelvic_axis/norm(sagittal_pelvic_axis));
        if(cross_product(3)>0)
            pelvic_rotation=-pelvic_rotation;% positive value for clockwise roration
        end

        %calculate Hip flexion/extension (right)
        hip_fexion_axis=transverse_pelvic_axis;
        v1_thigh_r=mean(data_seg(find(contains(channels,'R_FLE')),:)-data_seg(find(contains(channels,'R_FTC')),:),2);
        v2_thigh_r=mean(data_seg(find(contains(channels,'R_FME')),:)-data_seg(find(contains(channels,'R_FTC')),:),2);
        normal_vector_thigh_r=cross(v1_thigh_r,v2_thigh_r);
        [v1,v2]=get_perpendicular_basis_vector(hip_fexion_axis);
        p_vector1=project2plane(normal_vector_thigh_r,v1,v2);
        p_vector2=project2plane(sagittal_pelvic_axis,v1,v2);
        hip_flexion_r=acos(dot(p_vector1/norm(p_vector1),p_vector2/norm(p_vector2)));
        cross_product=cross(cross(p_vector1,p_vector2),forward_direction);
        if(cross_product(3)>0)
            hip_flexion_r=-hip_flexion_r; % positive value for flexion, and negative value for extension
        end


        %calculate hip abduction/adduction (right)
        hip_joint_r=mean(data_seg(find(contains(channels,'R_FTC')),:),2);
        knee_joint_center_r=mean(data_seg(find(contains(channels,'R_FLE')),:)+data_seg(find(contains(channels,'R_FME')),:),2)/2;
        thigh_long_axis_r=knee_joint_center_r-hip_joint_r;
        thigh_long_axis_r_length=norm(thigh_long_axis_r);
        v1=knee_joint_center_r-(hip_joint_r+thigh_long_axis_r_length*hip_fexion_axis/norm(hip_fexion_axis));
        v2=knee_joint_center_r-(hip_joint_r-thigh_long_axis_r_length*hip_fexion_axis/norm(hip_fexion_axis));
        normal_vector_pelvis_plane=-cross(transverse_pelvis_plane_v1,transverse_pelvis_plane_v2);
        p_vector1=project2plane(thigh_long_axis_r,v1,v2);
        p_vector2=project2plane(normal_vector_pelvis_plane,v1,v2);
        hip_adduction_r=acos(dot(p_vector1/norm(p_vector1),p_vector2/norm(p_vector2)));
        cross_product=cross(p_vector1,p_vector2);
        angle=acos(dot(cross_product/norm(cross_product),forward_direction/norm(forward_direction)));
        if(angle<pi/2)
            hip_adduction_r=-hip_adduction_r; % positive value for adduction (inwardly moved), and negative value for abduction
        end

        %calculate hip rotation (right)
        [v1,v2]=get_perpendicular_basis_vector(thigh_long_axis_r);
        p_vector1=project2plane(normal_vector_thigh_r,v1,v2);
        p_vector2=project2plane(sagittal_pelvic_axis,v1,v2);
        hip_rotation_r=acos(dot(p_vector1/norm(p_vector1),p_vector2/norm(p_vector2)));
        cross_product=cross(p_vector1,p_vector2);
        if(cross_product(3)>0)
            hip_rotation_r=-hip_rotation_r; % positive value for internal rotation
        end

        %calculate knee flexion/extension (right)
        knee_flexion_axis_r=mean(data_seg(find(contains(channels,'R_FLE')),:)-data_seg(find(contains(channels,'R_FME')),:),2);
        [v1,v2]=get_perpendicular_basis_vector(knee_flexion_axis_r);
        ankle_joint_r=mean(data_seg(find(contains(channels,'R_LM')),:),2);
        v1_shank_r=ankle_joint_r-mean(data_seg(find(contains(channels,'R_FLE')),:),2);
        v2_shank_r=ankle_joint_r-mean(data_seg(find(contains(channels,'R_FME')),:),2);
        normal_vector_shank_r=-cross(v1_shank_r,v2_shank_r);
        p_vector1=project2plane(normal_vector_shank_r,v1,v2);
        p_vector2=project2plane(normal_vector_thigh_r,v1,v2);
        knee_flexion_r=acos(dot(p_vector1/norm(p_vector1),p_vector2/norm(p_vector2)));

        %calculate ankle dorsi/plantarflexion (right)
        foot_toe_r=mean(data_seg(find(contains(channels,'R_MH1')),:),2);
        foot_vector_r=foot_toe_r-ankle_joint_r;
        [v1,v2]=get_perpendicular_basis_vector(knee_flexion_axis_r);
        p_vector1=project2plane(foot_vector_r,v1,v2);
        p_vector2=normal_vector_shank_r;
        ankle_dorsiflexion_r=acos(dot(p_vector1/norm(p_vector1),p_vector2/norm(p_vector2)));
        cross_product=cross(cross(p_vector1,p_vector2),forward_direction);
        if(cross_product(3)>0)
            ankle_dorsiflexion_r=-ankle_dorsiflexion_r; % positive value corresponds to dorsiflexion
        end

        %calculate foot progression
        transverse_laboratory_plane_v1=[1,0,0];
        transverse_laboratory_plane_v2=[0,1,0];
        p_vector=project2plane(foot_vector_r,transverse_laboratory_plane_v1,transverse_laboratory_plane_v2);
        foot_progression_r=acos(dot(p_vector/norm(p_vector),forward_direction/norm(forward_direction)));
        cross_product=cross(p_vector,forward_direction);
        if(cross_product(3)>0)
            foot_progression_r=-foot_progression_r;
        end


        feature_window=[pelvic_tilt,pelvic_obliquity,pelvic_rotation,hip_flexion_r,hip_adduction_r,hip_rotation_r,knee_flexion_r,ankle_dorsiflexion_r,foot_progression_r];
        feature_tmp(idx_window,:)=feature_window;
    end
    feature(:,:,i)=feature_tmp;
end



