function LAPC = Laplacian_centrality(A,norm)

% Calculates laplacian centrality for a weighted or unweighted undirected
% network. Can give the absolute or relative change in laplacian energy

if nargin < 2
    norm = 0;
end


N = length(A);
LAPC = zeros(1,N);


if length(unique(A)) == 2 && max(max(A)) == 1
    deg = sum(A);
for i = 1:N
       % Find the neighbours of node i
       nei = (A(i,:) > 0);      
       LAPC(i) = deg(i)^2 + deg(i) + 2*sum(deg(nei));    
end


else
    
for i = 1:N

NWC = sum(A(i,:).^2);

    nei = find(A(i,:) > 0);  
    NWE = 0;
    for j = 1:length(nei)
        NWE = NWE + sum(A(i,nei(j))*A(:,nei(j))) - A(i,nei(j))*A(i,nei(j));       
    end

    NWM = 0;
    for j = 1:length(nei)-1
        for k = j+1:length(nei)
           NWM = NWM + A(i,nei(j))*A(i,nei(k));
        end
    end
  
LAPC(i) = 2*NWE + 2*NWM + 4*NWC;



end

end

if norm
    LAP = diag(sum(A)) - A;
    LAP_ENERGY = sum(eig(LAP).^2);
    LAPC = LAPC/LAP_ENERGY;
end
