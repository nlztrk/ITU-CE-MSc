function [bc,c,b] = bridging_centrality(A,b)

if nargin < 2
    [b]=betweenness_bin(A);
end
n=length(A);
d=sum(A);  
if max(b) > 1
    b=b./((n-1)*(n-2));
end
c=zeros(size(b));

% degree;
G = logical(A);

for k=1:n
    c(k)=(1./d(k))./sum(1./d(G(:,k)));
end


bc=b.*c;
