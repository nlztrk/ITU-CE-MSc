function [RWC,H] = random_walk_centrality(adj)
% Calculates random walk closeness centrality. This can be interpreted as
% the average amount of time it takes a random walker to reach node i from
% any other node in the network

% INPUTS:                   adj = an N*N adjacency matrix where adj(i,j)
%                                 indicates i is the source and j is the
%                                 target
%
% OUTPUTS:                  RWC = the random walk closeness of each node
%
%                           H   = the mean first passage time of the
%                                 network. H(i,j) indicates the average
%                                 number of steps a random walker takes to
%                                 reach node j from i
%
% Stuart Oldham, Monash University, 2017

% Make markov chain 
M = diag(1./sum(adj,2))*adj;

n = length(M);

% Checks if the transition matrix contains no absorbing states and is 
% irreducible
if max(diag(M)) == 1
    error('The transition matrix contains an absorbing state!')  
end
if max(isnan(M)) == 1
    error('The transition matrix is not irreducible!')
end
comp = graphComponents(M);
if sum(comp) ~= length(comp)
    error('The transition matrix is not irreducible!')
end

% Make identity matrix
I = eye(n);

% As sM=s, it can also be written s(M-I)=O or sA=O (O is a 1*n null
% matrix). There are many, many different possible solutions so an extra 
% equation is needed to find a unique solution. As sum(s)=1, this can be 
% substituted into sA=O by replacing column i in A with 1s as well as 
% replacing element i in O with 1. A unique solution (i.e. the steady 
% state) can now be obtained.
A = M-I;

% Sets the last column of A to 1s so a unique solution can be obtained
A(:,end) = ones(n,1);

% Creates a vector of zeros apart from the last element is 1
O = zeros(1,n);
O(end) = 1;

% Find the steady state
s = O/A;

% Creates a matrix S where each row is a repeat of the array s element 
% (s(i) = S(:,i))
S = repmat(s,n,1); 

% Compute the fundamental matrix Z
Z = inv(I-M+S);

% Instead of running a for loop of H(i,j) = (Z(j,j) - Z(i,j))/s(j) this
% creates a matrix where the diagonal of Z (expressed as an array) is 
% repeated on each row (thus element (:,j) in this matrix is Z(j,j)). Z can
% then be subtracted from this matrix and each element will now be
% (Z(j,j) - Z(i,j)). This can now be divided on an elemental basis as
% S(:,j) = s(j) 
H = (repmat(diag(Z)',n,1)-Z)./S; 

% Equation 5 in Blochl et al., 2011
RWC = n./sum(H);