function c = closeness_bin(A,alt)

% This function calculates unweighted closeness centrality of a network A
% using functions from the Brain Connectivity Toolbox.
%
% Inputs:                               A = adjecency matrix
%                                     alt = alternative version of
%                                           closeeness centrality (nodal
%                                           efficiency)
% 
% Output:                               C = closeness centrality
%
% Stuart Oldham, Monash University, 2017
if nargin < 2
    alt = 0;
end

bin = double(A > 0);
n = length(bin);

if ~alt
    c = n./sum(distance_bin(bin));
else
    c = nodal_efficiency(A);
end

end