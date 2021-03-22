function Q = modularity_q(A,c)

% This funciton calculates the modularity index Q for a network A with a
% module membership of c. Adapted from the Brain Connectivity Toolbox
% function modularity_und
%
% Stuart Oldham, Monash University, 2018

n = length(A);
k = sum(A);
m = sum(k);
B = A - (k.'*k)/m;
s = c(:,ones(1,n));
Q = ~(s-s.').*B/m;
Q = sum(Q(:));