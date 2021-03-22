function cinf = information_centrality(A,alt)

% This function calculates information centrality on an unweighted/weighted
% network.
%
% Inputs:                               A = an unweighted/weighted
%                                           adjacency matrix
%                                     alt = an alternative version of
%                                           information centrality (see
%                                           below)
% Output:                            cinf = a vector of each nodes
%                                           information centrality
%
% As shown in https://pdfs.semanticscholar.org/0eb8/54ff50f1b647867c5d59d8ade4690ac8ea2c.pdf, 
% the same rank order of nodes by information centrality can be produced by
% the equation 1./diag(C) becuase all other paths are constants 
%
% Stuart Oldham, Monash University, 2017

if nargin < 2
    alt = 0;
end

% Obtain the number of nodes
n = length(A);

% Create a diagonal matrix where the value D(i,i) is the degree of node i 
D = diag(sum(A,2));

L = D-A;
J = ones(n);

C = (L + J)^-1;

switch alt
    case 1
        cinf = 1./diag(C);
    case 0
        Cdiag = diag(C);
        T = sum(Cdiag);
        RR = sum(C,2);
        cinf = (Cdiag(1:n) + (T - 2*RR(1:n))./n).^-1;
% How you would calculate it if you are using loops        
%         T = sum(diag(C));
%         cinf = zeros(n,1);
%         for u = 1:n
%             R = sum(C(u,:));
%             cinf(u) = inv(C(u,u) + (T - 2*R)/n);
%         end
end

end