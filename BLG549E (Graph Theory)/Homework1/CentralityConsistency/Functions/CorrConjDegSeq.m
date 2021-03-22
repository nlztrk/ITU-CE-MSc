function [dk,d] = CorrConjDegSeq(degseq)

% This function calculate the corrected conjugated sequence. For a node in
% position k in the degree sequence, the conjugated sequence describes how 
% many nodes before it in the degree sequence have a degree greater than 
% or equal to k-1, and how many nodes following it in the degree sequence 
% have a degree greater than or equal to k

% Input:                            degseq = degree sequence
%
% Outputs:                              dk = corrected conjugated sequence
%                                        d = degree sequence ordered from
%                                            highest to lowest
%
% Stuart Oldham, Monash University, 2017

d = sort(degseq,'descend');
n = length(d);
dk = zeros(1,n);
for k = 1:n
   dk(k) = sum(d(1:k - 1) >= k-1) + sum(d(k+1:n) >= k);
end
