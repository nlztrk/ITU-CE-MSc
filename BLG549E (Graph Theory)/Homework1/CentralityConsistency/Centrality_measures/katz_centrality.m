function c_Katz = katz_centrality(adj,a,b)

% This function computes Katz centrality for a network
%
% inputs:                   adj = an N*N adjacency matrix where adj(i,j)
%                                 indicates i is the source and j is the
%                                 target
%                           a   = alpha value (optional, defaults to .9 of
%                                 the largest eigenvalue
%                           b   = beta value (optional, defaults to 1)
% 
% output:                c_Katz = a vector of each nodes Katz Centrality
%
% Stuart Oldham, Monash University, 2017

A = full(double(adj > 0));
n = length(A);
invmaxeig = 1/max(eig(A));

if nargin < 2
   % a is set to be 10% less than the inverse of the largest eigenvalue
   a = .9*invmaxeig;
   % fprintf('a has been set to %.3f\n',a)
end

if nargin < 3
    b = 1;
end

if a <= 0
    error('a must be greater than zero or all nodes will have the same value as b')
elseif a >= invmaxeig
    error('a needs to be less than the inverse of the largest eigenvector (%.3f) of A',invmaxeig)
end


AT = A';

I_vec = b*ones(n,1);
I = diag(I_vec);
c_Katz = (inv(I - a*AT))*I_vec;

end