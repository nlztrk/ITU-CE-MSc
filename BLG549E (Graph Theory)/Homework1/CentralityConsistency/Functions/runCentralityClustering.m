function [Z, D, Hclusters, DB_vals] = runCentralityClustering(dataMatrix,NumClust)

if nargin < 2
    NumClust = 50;
end

if length(dataMatrix) < 50
    NumClust = size(dataMatrix,1);
end

Hclusters = zeros(size(dataMatrix,1),NumClust);

Z = linkage(dataMatrix,'ward','euclidean');

Y = pdist(dataMatrix,'euclidean');
D = squareform(Y);

for x = 1:NumClust
  Hclusters(:,x) = cluster(Z,'MaxClust',x); 
end   

hevaDB_d = evalclusters(dataMatrix,Hclusters,'DaviesBouldin');

DB_vals = hevaDB_d.CriterionValues;