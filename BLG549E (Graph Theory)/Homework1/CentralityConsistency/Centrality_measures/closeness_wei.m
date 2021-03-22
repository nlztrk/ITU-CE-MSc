function c = closeness_wei(A,invert,alt)

% This function calculates weighted closeness centrality of a network A
% using functions from the Brain Connectivity Toolbox.
%
% Inputs:                               A = adjecency matrix
%                                  invert = invert matrix weights
%                                     alt = alternative version of
%                                           closeeness centrality (nodal
%                                           efficiency)
% 
% Output:                               C = closeness centrality
%
% Stuart Oldham, Monash University, 2017
if nargin < 2
    invert = 0;
end

if nargin < 3
    alt = 0;
end

A = sparse(A);

n = length(A);
if invert && ~alt
    %c = n./sum(distance_wei(1./A));
    c = n./sum(graphallshortestpaths(1./A));
elseif ~invert && ~alt
    %c = n./sum(distance_wei(A));
    c = n./sum(graphallshortestpaths(A));
elseif invert && alt
    %c = nodal_efficiency(1./A);
    c = n./sum(graphallshortestpaths(A));
elseif ~invert && alt
    %c = nodal_efficiency(A);
    c = n./sum(graphallshortestpaths(A));
end

end