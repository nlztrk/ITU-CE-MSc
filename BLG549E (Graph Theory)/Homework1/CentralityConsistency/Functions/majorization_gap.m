function [tmg,tmg_norm] = majorization_gap(d)

% Calculates the majorization gap for a given network or degree sequence.
% The majorization gap is an estimate of the number of edges that would
% need to be rewired to transform the network/degree sequence to that of a
% threshold graph. The normalised version of the majorization gap is simply
% the number of estimated rewirings divided by the number of edges
%
% Input:                                d = degree sequence or an
%                                           undirected adjacency matrix
%
% Outputs:                            tmg = majorization gap
%                                tmg_norm = normalised majorization gap
%
% Stuart Oldham, Monash University, 2017

    if ~isvector(d)
       bin = double(d > 0);
       d = sum(bin);
    end
    [CorrConjSeq,dsort] = CorrConjDegSeq(d);
    x = CorrConjSeq - dsort;
    
    tmg = .5*sum(x(x>0));
    edges = sum(d)/2;
    tmg_norm = tmg/edges;
end