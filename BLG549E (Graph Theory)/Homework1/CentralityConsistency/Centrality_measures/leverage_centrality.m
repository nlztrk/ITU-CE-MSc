function l = leverage_centrality(adj)
% This function computes leverage centrality. Is used for undirected
% networks and the measure can be weighted or unweighted (simply change the
% input adjacency matrix to be binary or not)
%
% Input:                adj = adjacency matrix
%
% Output:               l = a vector containing each node's leverage
%                       centrality
%
% Stuart Oldham, Monash University, 2017

str = sum(adj);
l = zeros(1,length(adj));
for i = 1:length(adj)
   Ni = str((adj(:,i) > 0));
   % The leverage centrality equation
   l(i) = mean((str(i) - Ni)./(str(i) + Ni));
end