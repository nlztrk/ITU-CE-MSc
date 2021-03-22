function b = random_walk_betweenness(A,parallel)
% This function computes Newman's random-walk betweenness centrality
% measure. Conceptually, the random-walk betweenness centrality of node i
% is equal to the number of times a random walker which begins at node s
% and ends at node t passes through node i along the way, averaged over all
% s and t.
%
% This measure is only currently working for unweighted, undirected
% networks. 
%
% Inputs:                               A = adjacency matrix
%                                parallel = set to 1 to use MATLABS 
%                                           parallel processing toolbox (if
%                                           installed)                    
%
% Output:                               b = a vector containing each nodes
%                                           random-walk betweenness 
%                                           centrality
%
% Stuart Oldham, Monash University, 2017

% Check if parallel toolbox is installed
if ~license( 'test', 'Distrib_Computing_Toolbox' )
    warning('Parallel processing toolbox not installed') 
    parallel = 0;
end

% Obtain the number of nodes
n = length(A);

% Create a diagonal matrix where the value D(i,i) is the degree of node i 
D = diag(sum(A,2));

% The matrix T is formed by removing the last row and column from matrix A
% and D then subtracting A from D before inversing this matrix
ind = 1:n-1;
T=inv(D(ind,ind)-A(ind,ind));

% The removed row and column is added back in with all values equal to zero
T(n,:) = zeros(1,n-1);
T(:,n) = zeros(n,1);

% Initialise the vector I. I(i) is sum of the current flowing through
% node i for a given source, s, and target, t
I = zeros(1,n);

neighbours = cell(n,1);

for i = 1:n
    neighbours{i} = find(A(i,:) > 0);
end
fprintf('Starting RWBC\n')
if parallel
I = num2cell(I);
        % Loop over every node to calculate the current flowing through it
        % for the given s and t
    parfor i = 1:n
        tic
        for s = 1:n-1
            for t = s+1:n
                if i == s || i == t
                    % Equation 10 in Newman, 2003
                    I{i} = I{i} + 1;
                else
                    % j is the neighbours of node i
                    j = neighbours{i};
                    % Equation 9 in Newman, 2003
                    I{i} = I{i} + .5*sum(A(i,j)*abs(T(i,s)-T(i,t)-T(j,s)+T(j,t)));
                end
            end
        end
        toc
    end
    I = cell2mat(I);
else
% As the same result will be obtained for s = i, t = j and s = j, t = i,
% lood over each source-target pair
for s = 1:n-1
    for t = s+1:n
        % Loop over every node to calculate the current flowing through it
        % for the given s and t
        for i = 1:n
            if i == s || i == t
                % Equation 10 in Newman, 2003
                I(i) = I(i) + 1;
            else
                % j is the neighbours of node i
                j = neighbours{i};
                % Equation 9 in Newman, 2003
                %I(i) = I(i) + .5*sum(arrayfun(@(x) A(i,x)*abs(T(i,s)-T(i,t)-T(x,s)+T(x,t)),j));
                I(i) = I(i) + .5*sum(A(i,j)*abs(T(i,s)-T(i,t)-T(j,s)+T(j,t)));
            end
        end
    end
end

end

% Equation 11 in Newman, 2003
b = I/(.5*n*(n-1));

end
