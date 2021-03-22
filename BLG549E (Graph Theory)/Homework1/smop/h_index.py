# Generated with SMOP  0.41-beta
from libsmop import *
# h_index.m

    
@function
def h_index(input_=None,*args,**kwargs):
    varargin = h_index.varargin
    nargin = h_index.nargin

    # Calculates each nodes h-index (i.e. a node with an index h has h
# neighbours with have a degree/strength of h).
    
    # Input:               input = an undirected adjacency matrix (can be
#                              weighted or unweighted) or a vector
    
    # Output:                  h = a vector containing each nodes h index
    
    # Calculate strength (for a weighted network) or degree (for an unweighted 
# network)
    
    # Stuart Oldham, Monash University, 2017
    
    # If the input is a vector calculate the h-index of that
    if isvector(input_):
        num=length(input_)
# h_index.m:17
        h_vals=zeros(1,num)
# h_index.m:18
        for j in arange(1,num).reshape(-1):
            h_vals[j]=length(find(input_ >= j))
# h_index.m:21
        # Calculates the h-index
        h=max(min(arange(1,num),h_vals))
# h_index.m:24
        # If the input is a, adjacency matrix calculate the h-index of that
    else:
        if ismatrix(input_):
            adj=copy(input_)
# h_index.m:27
            str=sum(adj)
# h_index.m:28
            h=zeros(1,length(str))
# h_index.m:30
            for i in arange(1,length(adj)).reshape(-1):
                # Find the neighbours of node i
                nei=str((adj(i,arange()) > 0))
# h_index.m:33
                num_nei=length(nei)
# h_index.m:34
                h_vals=zeros(1,num_nei)
# h_index.m:35
                # neighbours) and counts how many neighbours have at least j
       # degree/strength
                for j in arange(1,num_nei).reshape(-1):
                    h_vals[j]=length(find(nei >= j))
# h_index.m:40
                # Finds the last point at which h of node i's neighbours do not have 
       # less than h degree/strength. This is the nodes h-index
                h[i]=max(min(arange(1,num_nei),h_vals))
# h_index.m:44
        else:
            error('Unrecognised input')
    