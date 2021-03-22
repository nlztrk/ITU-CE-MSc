function [M, P, Q] = run_modularity(adj,nMod,tau)

% Runs a consensus approach to achieve determine a networks modularity
%
% Input:                            adj = adjacency mnatrix
%                                  nMod = number of iterations
%                                   tau = tau value
%
% Output:                             M = consensus module assignment
%                                     P = participation coefficient
%                                     Q = Modularity Q
%
% Stuart Oldham, Monash University, 2017

Q = zeros(1,nMod);

M_temp = zeros(length(adj),nMod);

for i = 1:nMod

    [M_temp(:,i), Q(i)] = community_louvain(adj);
    %fprintf('Completed %d\n',i)
end

D = agreement_weighted(M_temp,Q);

M = consensus_und(D,tau,nMod);
P = participation_coef(adj,M);