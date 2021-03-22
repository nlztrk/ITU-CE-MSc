# Generated with SMOP  0.41-beta
from libsmop import *
# bridging_centrality.m

    
@function
def bridging_centrality(A=None,b=None,*args,**kwargs):
    varargin = bridging_centrality.varargin
    nargin = bridging_centrality.nargin

    if nargin < 2:
        b=betweenness_bin(A)
# bridging_centrality.m:4
    
    n=length(A)
# bridging_centrality.m:6
    d=sum(A)
# bridging_centrality.m:7
    if max(b) > 1:
        b=b / (dot((n - 1),(n - 2)))
# bridging_centrality.m:9
    
    c=zeros(size(b))
# bridging_centrality.m:11
    # degree;
    G=logical(A)
# bridging_centrality.m:14
    for k in arange(1,n).reshape(-1):
        c[k]=(1.0 / d(k)) / sum(1.0 / d(G(arange(),k)))
# bridging_centrality.m:17
    
    bc=multiply(b,c)
# bridging_centrality.m:21