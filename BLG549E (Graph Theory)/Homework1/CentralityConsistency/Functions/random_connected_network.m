function [A,Edges] = random_connected_network(N,E,W,quiet)

% This function generates a random graph that will be fully connected. It
% first generates a random spanning tree of N -1 edges (i.e. a minimum 
% spanning tree). Secondly, it then adds in edges at random until the 
% desired number is generated. 
%
% The random spanning tree is generated uniformly from the set of possible 
% spanning tress. This is done by starting a random walk at a given node. 
% This random walk can travel to any other node. Each time the random walk 
% encounters a new node, an edge is added into the network between the new 
% node and the previous node. This continues until all nodes have been
% encountered.
%
% The output network should have properties very close to a random graph.
% Ideally use this code if the desired network is below the critical  
% threshold (i.e. log(N)/(N)). Otherwise simply generate the network on a
% probabilistic basis (will be faster, especially for larger networks)
%
% This is an impliamentation of the idea described here: https://stackoverflow.com/a/14618505

% Input:        N                   Number of nodes or a network
%               E                   Number of edges or desired density
%               W                   A vector of edge weights (the length of
%                                   the vector must match the number of
%                                   edges requested)
%               quiet               Set to 1 to turn off any messgaes the
%                                   code outputs
%
% Output:       A                   A fully connected random network
% 
%
% Stuart Oldham, Monash University, 2017

if ~ismatrix(N) && length(N) > 1
    error('Unrecognised input for N')
end

if nargin < 3
    W = [];
end

if nargin < 4
    quiet = 0;
end

% Check if the input network is weighted. If it is, the weighted version 
% will be run. Any other inputs will be ignored and overridden  

if length(N) > 1
    CIJ = N;
    E = nnz(triu(CIJ,1)); 
    N = length(CIJ);
    if length(unique(CIJ)) ~= 2
        triu_CIJ = triu(CIJ,1);
        if ~isempty(W)
            warning('Input matrix is weighted and a vector of weights was also provided. Defaulting to the input network weights')            
        end
        W = CIJ(triu_CIJ > 0);
    else
        W = [];
    end
end

if length(E) > 1
    error('Unrecognised input for E')
end

if isempty(W)
    runWeighted = 0;
else
    runWeighted = 1;
end

% Get the number of edges to create.

if E <= 1 && E > 0
    Edges = round(E * ((N^2 - N) /2));
elseif E > 1
    Edges = E;
else
    error('E must be greater than 0')
end

% Check if the number of edges is enough to make a fully connected network

if Edges < N - 1
    error('Unable to make a fully connected network')
end

% Check if the number of edges and weights match

if runWeighted
    if length(W) ~= Edges
       error('%d edges requested but only %d weights were supplied',Edges,length(W)) 
    end
end

% Initalize a vector of the index nodes not currently in the MST

Nodes_out_of_MST = 1:N;

% Chose a random node to start on

current_node = randi(N,1);

% Remove the start node from the vector of nodes not currently in the MST

Nodes_out_of_MST(current_node) = [];

% Initalize the MST

MST = zeros(N);

% This is just a bit of code used for printing the progression of the code

reverseStr = '';

% Initalize a scalar of nodes remaining out of the MST

Nodes_left = N - 1;
while ~isempty(Nodes_out_of_MST)
    
    % Create a vector of all nodes except the current node

    Nei = 1:N;
    Nei(current_node) = [];
    
    % Choose the next random node
    
    next_node = Nei(randi(N-1,1));
    
    % If the next randomly chosen node isn't currently part of the MST, add
    % it in. Otherwise try again
    
    if ismember(next_node,Nodes_out_of_MST)
        % Add a connection between the current node and the next node
        MST(current_node,next_node) = 1;
        MST(next_node,current_node) = 1;
        % Remove the node from the vector of nodes not currently in the MST
        Nodes_out_of_MST(Nodes_out_of_MST == next_node) = [];
        % Reduce the number of nodes remaining by 1
        Nodes_left = Nodes_left - 1;
    end
    if ~quiet
        msg = sprintf('%d/%d nodes still to be assigned to MST\n', Nodes_left, N);
        fprintf([reverseStr, msg]);
        reverseStr = repmat(sprintf('\b'), 1, length(msg));
    end
    % Make the next node the current node for the next interation
    current_node = next_node;      
end

% Create a matrix of random numbers and add the MST onto that

r = rand(N) + MST;

% Make the matrix symmetrical. This is so that each unique edge has a
% single value associated to it

R = triu(r,1) + triu(r,1)';

% Get a vector of the values and positon of the upper triangle elements in
% R. Then sort these values from largest to smallest and the respective
% ordering

[R_vec,R_vec_ord] = triu2vec(R,1);

[~,ind] = sort(R_vec,'descend');

% Find the index of edges to include. This will result in exactly E edges
% being included in the network. You could make it so all edges above/below
% a predefined probability are included, similar to how random networks are
% typically created but given this code is forcing a minimum number of
% edges to be generated I think adding in a specific number of edges is
% more appropriate

IncludedEdges = R_vec_ord(ind(1:Edges));

% Set all included edges to 1 and make the matrix symmetrical

A = zeros(N);

A(IncludedEdges) = 1;

% Make the matrix symmetrical

A = A + A';

if runWeighted

    % Extract the upper triangle of A
    
    triuA = triu(A,1);
    
    % Generate a vector of length E of random numbers and order it
    
    [~,rand_ord] = sort(rand(E,1));

    % Randomise the order of the weights in the weight vector
    
    W_randord = W(rand_ord);
    
    % Apply the randomised weights to a given edge in triuA

    triuA(triuA == 1) = W_randord;
    
    % Make the matrix symmetrical

    A = triuA + triuA';

end

end

function [vec,ind] = triu2vec(mat,k)
    onesmat = ones(size(mat));
    UT = triu(onesmat,k);
    vec = mat(UT == 1);
    ind = find(UT == 1);
end