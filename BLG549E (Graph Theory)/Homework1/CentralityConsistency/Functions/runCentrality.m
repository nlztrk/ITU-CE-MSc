function [centrality,Q,centrality_names,centrality_names_abbrev] = runCentrality(Network,weighted,runparallel,quiet,exclude)

% This script runs a number of different centrality measures for a network
% or set of networks.
%
% Input:                   Network = an adjacency matrix or a cell where
%                                    each cell is an adjacency matrix or a
%                                    3D matrix where the third dimension is
%                                    an individual network
%                         weighted = set to 0 for unweighted centrality 
%                                    measures for weighted networks, invert the
%                                    weights for measures which assume a
%                                    higher edge weight value indicates 
%                                    less importance (no effect on 
%                                    unweighted networks).
%                      runparallel = run in parallel if set to 1 (default
%                                    is 0). Usefully when you have very
%                                    large networks (> 1000 nodes)
%
% Output:               centrality = matrix of centrality scores or a cell
%                                    array where each cell contains a 
%                                    matrix of centrality scores
%                                Q = the Q value for the modularity of the
%                                    network
%                 centrality_names = name of each centrality measure   
%          centrality_names_abbrev = abbreviated name of each centrality 
%                                    measure 
%
% The script will detect if the network is unweighted or weighted so it 
% should be in that format before being passed to this function. 
%
% Stuart Oldham, Monash University, 2018

if nargin < 2
    weighted = 0;
end

if nargin < 3
    runparallel = 0;
end

if nargin < 4
    quiet = 0;
end

if nargin < 5
   exclude = []; 
end

if ~iscell(Network)
    [x,y,z] = size(Network);
    if z > 1
        NetCell = squeeze(mat2cell(Y, x, y,[1 1 1]));
    else
        NetCell{1} = Network; 
    end
else
    NetCell = Network;
end

NumNets = length(NetCell);
Q = zeros(1,NumNets);
if NumNets > 2
    centrality = cell(1,NumNets);
end
    
for i = 1:NumNets
    adj = NetCell{i};
    NumNodes = length(adj);
    c = zeros(17-length(exclude),NumNodes);
    
    if weighted == 2
       adj_inv = 1./adj;
       adj_inv(adj_inv == inf) = 0;
    elseif weighted == 0
        adj_inv = double(adj>0);
        adj = double(adj>0);
    elseif weighted == 1
        adj_inv = adj;
    end
    
    if weighted == 0
        wei = 0;
    else
        wei = 1;
    end
    idx = 1;
    if ~ismember(1,exclude); c(idx,:) = strengths_und(adj); idx = idx + 1; end
    if ~ismember(2,exclude); c(idx,:) = betweenness_centrality(sparse(adj_inv)); idx = idx + 1; end
    if ~ismember(3,exclude); c(idx,:) = eigenvector_centrality_und(full(adj)); idx = idx + 1; end 
    if ~ismember(4,exclude); c(idx,:) = pagerank_centrality(adj,.85); idx = idx + 1; end
    if ~ismember(5,exclude); c(idx,:) = closeness_wei(full(adj_inv)); idx = idx + 1; end
    if ~ismember(6,exclude); c(idx,:) = diag(communicability(full(adj),wei,'network')); idx = idx + 1; end
    if ~ismember(7,exclude); c(idx,:) = random_walk_centrality(adj); idx = idx + 1; end
    if ~ismember(8,exclude); c(idx,:) = h_index(adj); idx = idx + 1; end
    if ~ismember(9,exclude); c(idx,:) = leverage_centrality(adj); idx = idx + 1; end
    if ~ismember(10,exclude); c(idx,:) = information_centrality(adj); idx = idx + 1; end 
    if ~ismember(11,exclude); c(idx,:) = katz_centrality(adj); idx = idx + 1; end
    if ~ismember(12,exclude); c(idx,:) = communicability(full(adj),wei,'nodal'); idx = idx + 1; end
    if ~ismember(13,exclude); c(idx,:) = random_walk_betweenness(adj,runparallel); idx = idx + 1; end
    if ~ismember(14,exclude); c(idx,:) = communicability_betweenness(full(adj),wei,runparallel); idx = idx + 1; end
    if ~ismember(15,exclude)
        [M, c(idx,:)] = run_modularity(adj,50,.4); 
        idx = idx + 1; 
        Q(i) = modularity_q(adj,M);
    else
        Q(i) = NaN;
    end
    if ~ismember(16,exclude); c(idx,:) = Laplacian_centrality(full(adj)); idx = idx + 1; end
    if ~ismember(17,exclude); c(idx,:) = bridging_centrality(full(adj),c(2,:)); idx = idx + 1; end
    
    % The Q value is calculated on the consensus partition
    
    
    
    if NumNets > 2
        centrality{i} = c;
        if ~quiet
           fprintf('Completed centrality analysis of network %d\n',i) 
        end
    else
        centrality = c;
    end
end

centrality_names{1} = 'strength'; centrality_names_abbrev{1} = 'DC';
centrality_names{2} = 'betweenness'; centrality_names_abbrev{2} = 'BC';
centrality_names{3} = 'eigenvector'; centrality_names_abbrev{3} = 'EC';
centrality_names{4} = 'pagerank'; centrality_names_abbrev{4} = 'PR'; 
centrality_names{5} = 'closeness'; centrality_names_abbrev{5} = 'CC'; 
centrality_names{6} = 'subgraph'; centrality_names_abbrev{6} = 'SC'; 
centrality_names{7} = 'random-walk closeness'; centrality_names_abbrev{7} = 'RWCC'; 
centrality_names{8} = 'h-index'; centrality_names_abbrev{8} = 'HC'; 
centrality_names{9} = 'leverage'; centrality_names_abbrev{9} = 'LC'; 
centrality_names{10} = 'information'; centrality_names_abbrev{10} = 'IC'; 
centrality_names{11} = 'katz'; centrality_names_abbrev{11} = 'KC';
centrality_names{12} = 'total communicability'; centrality_names_abbrev{12} = 'TCC';
centrality_names{13} = 'random-walk betweenness'; centrality_names_abbrev{13} = 'RWBC';
centrality_names{14} = 'communicability betweenness'; centrality_names_abbrev{14} = 'CBC';
centrality_names{15} = 'participation coefficient'; centrality_names_abbrev{15} = 'PC';
centrality_names{16} = 'Laplacian'; centrality_names_abbrev{16} = 'LAPC';
centrality_names{17} = 'Bridging'; centrality_names_abbrev{17} = 'BridC';

centrality_names(exclude) = [];
centrality_names_abbrev(exclude) = [];