function G = communicability(A,weighted,measure)
% COMMUNICABILITY(A) computes the communicability of pairs of nodes in the
% network represented by the adjacency matrix A. It returns either a matrix 
% whose elements G(i,j) = G(j,i) give the the communicability between nodes 
% i and j, the mean across all node pairs of this matrix or the row mean of 
% G
%
%   Inputs:                 A = undirected adjacency matrix
%
%                           weighted = 0 or 1. If 1 the weighted version of
%                           communicability will be calculated and if 0,
%                           the standard version will be used
%
%                           measure = 'network', 'global' or 'nodal'
%                                   'network' = the output G will be a
%                                   communicability network (default)
%                                   'global' = a value representing a
%                                   summary measure of the networks
%                                   communicability
%                                   'nodal' = a vector of each nodes mean
%                                   communicability
%
%   Output:                 G = the desired output specified by the
%                           variable 'measure'
%
% Stuart Oldham, Monash University, 2017
if nargin < 2
    weighted = 0;
end

if nargin < 3
    measure = 'network';
end

comp = graphComponents(A);
if sum(comp) ~= length(comp)
    error('The network is fragmented! Cannot calculate communicability')
end

if ~weighted
bin = double(A > 0);
    switch measure
        case 'network'
           G = gexpm(bin);
        case 'nodal'
           G = sum(gexpm(bin),2);
        case 'global'
           N = length(bin);
           g = gexpm(bin);
           EYE = logical(eye(N,N));
           G = mean(g(~EYE));
        otherwise
           error('Input for ''measure'' not recognised')
    end
else
   S = diag(sum(A,2));
   red_adj = (S^-.5)*A*(S^-.5);
    switch measure
        case 'network'
           G = gexpm(red_adj);
        case 'nodal'
           G = sum(gexpm(red_adj),2);
        case 'global'
           N = length(A);
           g = gexpm(red_adj);
           EYE = logical(eye(N,N));
           G = mean(g(~EYE));
        otherwise
           error('Input for ''measure'' not recognised')
    end
end