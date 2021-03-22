function h = h_index(input)
% Calculates each nodes h-index (i.e. a node with an index h has h
% neighbours with have a degree/strength of h).
%
% Input:               input = an undirected adjacency matrix (can be
%                              weighted or unweighted) or a vector
%
% Output:                  h = a vector containing each nodes h index

% Calculate strength (for a weighted network) or degree (for an unweighted 
% network)
%
% Stuart Oldham, Monash University, 2017

% If the input is a vector calculate the h-index of that
if isvector(input)
    num = length(input);
    h_vals = zeros(1,num);
    % Finds the number of values which exceed j
    for j = 1:num
           h_vals(j) = length(find(input >= j)); 
    end
    % Calculates the h-index
    h = max(min(1:num, h_vals));
% If the input is a, adjacency matrix calculate the h-index of that
elseif ismatrix(input)
    adj = input;
    str = sum(adj);

    h = zeros(1,length(str));
    for i = 1:length(adj)
       % Find the neighbours of node i
       nei = str((adj(i,:) > 0));
       num_nei = length(nei);
       h_vals = zeros(1,num_nei);
       % Loops over the neighbours of node i j times (where j is the number of
       % neighbours) and counts how many neighbours have at least j
       % degree/strength
       for j = 1:num_nei
           h_vals(j) = length(find(nei >= j)); 
       end
       % Finds the last point at which h of node i's neighbours do not have 
       % less than h degree/strength. This is the nodes h-index
       h(i) = max(min(1:num_nei, h_vals));
    end
else
    error('Unrecognised input')
end
