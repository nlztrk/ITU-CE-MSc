function [Enod,d] = nodal_efficiency(adj)

% Calculates nodal efficiency using the Brain Connectivity Toolbox
% functions
%
% Stuart Oldham, Monash University, 2017

d = distance_wei(adj);
N = length(adj);
d(d==0) = nan;
Enod = nansum(1./d) ./ (N-1);

end
