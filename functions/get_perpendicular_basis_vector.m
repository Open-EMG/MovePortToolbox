function [v1,v2]=get_perpendicular_basis_vector(normal_vector)
% calculate the basic vector of a plane perpendicular to a normal vector of
% the plane

x1=rand;
y1=rand;
z1=-(normal_vector(1)*x1+normal_vector(2)*y1)/normal_vector(3);
if(isinf(z1))
    z1=sign(z1)*max(abs(x1),abs(y1))*1000000000;
end
v1=[x1;y1;z1];
v1=v1/norm(v1);

v2=cross(normal_vector,v1);
if(size(v2,2)>1)
    v2=v2';
end
v2=v2/norm(v2);




