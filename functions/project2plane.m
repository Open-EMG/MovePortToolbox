function p_vector=project2plane(vector,v1,v2)
% v1 and v2 are two basis of the plane
% x is the original vector
% p_vector is the projected vector of x to the plane

if(size(vector,1)==1)
    vector=vector'; % turn vector to a column vector
end
V=[v1(1) v2(1);
    v1(2) v2(2);
    v1(3) v2(3)];
k=V'*V;
Ik=inv(k);
X=Ik*V'*vector;
p_vector=V*X;