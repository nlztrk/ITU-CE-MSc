% Main analysis script

% This script will rerun all analysis performed in Oldham et al., 2018. 
% Consistency and differences between centrality metrics across distinct 
% classes of networks.

% Running this script as is will be incredible computationally expensive
% and would likely take months running on a single computer. It would
% also require a significant amount of memory. Therefore
% running in parallel on multiple computers is recommended if trying to
% replicate (or use it for your own purposes)

% This code requires dependencies from the Brain Connectivity Toolbox

% Note that subtle differences may result if trying to recalculate the
% centrality measures as the calculation of the participation coefficient
% involves stochastic procedures. This differences will affect the
% correlations and clustering results but the differences will be very
% slight

%% Initial setup
% Define these variables for your own environment and desired parameters
% Define path to the directory of this script
MAINPATH = '/scratch/kg98/stuarto/CentralityConsistency-master';
% Define path to the directory of the BCT
BCTPATH = '/projects/kg98/stuarto/BCT';
% Define path to the MatlabBGL library
MATLABBGLPATH = '/scratch/kg98/stuarto/matlab_bgl';
% Define the number of nulls to generate
NumNulls = 100;
% Define the number of clusters to calculate
NumClust = 50;

%% Define paths and load networks

addpath(genpath(MAINPATH))
addpath(genpath(BCTPATH))
addpath(genpath(MATLABBGLPATH))

weighttype = 'Unweighted';
%weighttype = 'Weighted';

load([weighttype,'_networks.mat'])

NumNetworks = length(Networks);
%% Perform centrality measures on each real-world network

% Cent is the centrality scores for a network

% Q is the networks modularity

% cent_names is a cell array containing the name of each centrality measure

% cent_names_abbrev is a cell array containing the abbreviated name of each 
% centrality measure

% NetworkCentCorr is the CMCs for a network

% Network_mwCMC is the mean CMC for a network

NetworkPropsNames = {'Assortativity','Clustering','Density','Diffusion efficiency','Global efficiency','Majorization gap','Modularity','Spectral gap'};

for i = 1:NumNetworks

    A = Networks{i};
    
    [Cent,Q,cent_names,cent_names_abbrev] = runCentrality(A,0,1,0); 
    NetworkProps = full(calculateNetworkProperties(A));
    NetworkProps = [NetworkProps(1:6) Q NetworkProps(7)];
       
       
   NetworkCentCorr = corr(Cent','Type','Spearman'); 
   NetworkCentCorr(isnan(NetworkCentCorr)) = 0;
   Network_mwCMC = mean(triu2vec(NetworkCentCorr,1));
   
   save([weighttype,'_Network_',num2str(i),'_centrality_results.mat'],'-v7.3','A','Cent','Q','cent_names',...
    'cent_names_abbrev','NetworkProps','NetworkPropsNames','NetworkCentCorr','Network_mwCMC')

end


%% Create unconstrained nulls for each network. This step also generates the
% majorization gap for the unconstrained networks

% exclude sets the centrality measures to not run. 13 and 14 correspond to
% random walk betweenness and communicability betweenness, respectively.
% These were not run in the paper for the surrogates because the
% computation time required was too high :(

exclude = [13 14];

% ConNulls is the constrained surrogates for a network

% UnconNulls is the unconstrained surrogates for a network

% ConNullsCent is a cell of the centrality scores of the constrained
% nulls

% UnconNullsCent is a cell of the centrality scores of the unconstrained
% surrogates

% ConNullsQ is the modularity value for each constrained surrogate

% UnconNullsQ is the modularity value for each unconstrained surrogate

% ConNull_mwCMC s the mean within-network CMC for each constrained
% surrogate

% UnconNull_mwCMC s the mean within-network CMC for each unconstrained
% surrogate

for i = 1:NumNetworks
    A = Networks{i};
    
    ConNulls = cell(1,NumNulls);
    UnconNulls = cell(1,NumNulls);
    
    for j = 1:NumNulls
        ConNulls{j} = random_connected_network(A,[],[],1);
        UnconNulls{j} = make_ConstrainedNull(A,0);
    end

    
    [UnconNullsCent,UnconNullsQ] = runCentrality(UnconNulls,0,1,1,exclude); 
    [ConNullsCent,ConNullsQ,cent_names,cent_names_abbrev] = runCentrality(ConNulls,0,1,1,exclude); 

    UnconNullProps = zeros(NumNulls,8);
    ConNullProps = zeros(NumNulls,8);
    
    clear ConNullCentCorr ConNull_mwCMC UnconNullsCentCorr UnconNull_mwCMC
    
    for k = 1:NumNulls

        NullPropertys = full(calculateNetworkProperties(UnconNulls{k}));

        UnconNullProps(k,:) = [NullPropertys(1:6) UnconNullsQ(k) NullPropertys(7)];

        NullPropertys2 = full(calculateNetworkProperties(ConNulls{k}));

        ConNullProps(k,:) = [NullPropertys2(1:6) ConNullsQ(k) NullPropertys2(7)];
               
        % Sometimes each node can be assigned the same score in a
        % centrality measure. When calculating a correlation this returns a
        % NaN. We set these values to 0
        nullcorr = corr(ConNullsCent{k}','Type','Spearman');
        nullcorr(isnan(nullcorr)) = 0;
        ConNullCentCorr(:,:,k)=nullcorr;
            
        ConNull_mwCMC(k) = mean(triu2vec(nullcorr,1));
            
        nullcorr = corr(UnconNullsCent{k}','Type','Spearman');
        nullcorr(isnan(nullcorr)) = 0;
        UnconNullsCentCorr(:,:,k)=nullcorr;

        UnconNull_mwCMC(k) = mean(triu2vec(nullcorr,1));
        
    end
        load([weighttype,'_Network_',num2str(i),'_centrality_results.mat'],'Cent','NetworkProps');
        
        Cent(exclude,:) = [];

       NetworkCentCorr = corr(Cent','Type','Spearman'); 
       NetworkCentCorr(isnan(NetworkCentCorr)) = 0;
       Network_mwCMC = mean(triu2vec(NetworkCentCorr,1));
       
       
       save([weighttype,'_Network_',num2str(NetNumber),'_surrogate_results.mat'],'-v7.3','A','Cent','Q','cent_names','cent_names_abbrev',...
    'NetworkProps','NetworkPropsNames','UnconNullProps','ConNullProps',...
    'ConNullsCent','UnconNullsCent','NetworkCentCorr','Network_mwCMC','UnconNullsCentCorr','ConNullCentCorr','ConNull_mwCMC','UnconNull_mwCMC')

    save(['Unweighted_Network_',num2str(NetNumber),'_surrogates_adjmat.mat'],'-v7.3','ConNulls','UnconNulls')

       
end

%% Compile networks

NumNetworks = length(Networks);
NetworkProperties = zeros(NumNetworks,8);

% NetworksCentCorr is a 3D matrix of CMCs for each empirical network

NetworksCentCorr = zeros(15,15,NumNetworks);

% NetworksCentCorrCell is a cell of CMCs for each empirical network

NetworksCentCorrCell = cell(1,NumNetworks);
Networks_mwCMC = zeros(1,NumNetworks);
Type = zeros(1,NumNetworks);
NullNetworks_mwCMC = cell(NumNetworks,2);
NullNetProperty = cell(NumNetworks,2);
for i = 1:NumNetworks
   filename = [weighttype,'_Network_',num2str(i),'_surrogate_results.mat'];  
   load(filename)
    NetworkProperties(i,:) = NetworkProps;
    NetworksCentCorr(:,:,i) = NetworkCentCorr;
   NetworksCentCorrCell{i} = NetworksCentCorr(:,:,i); 
   Networks_mwCMC(i) = Network_mwCMC;
  switch NetworkType{i}
      case 'Biological'
         Type(i) = 1;
      case 'Social'
         Type(i) = 2;
      case 'Economic'
        Type(i) = 3;
     case 'Transportation'
        Type(i) = 4;
     case 'Technological'
        Type(i) = 5;
     case 'Informational'
        Type(i) = 6;
  end 

   NullNetworks_mwCMC{i,1} = UnconNull_mwCMC;
   NullNetworks_mwCMC{i,2} = ConNull_mwCMC;

   NullNetProperty{i,1} = UnconNullProps;
   NullNetProperty{i,2} = ConNullProps;
end

save(['Combined_',weighttype,'_surrogate_results.mat'],'NullNetworks_mwCMC','NullNetProperty','cent_names','cent_names_abbrev','Citations','NetworkNames','NetworkProperties','NetworkPropsNames','Networks','Networks_mwCMC','NetworksCentCorr','NetworksCentCorrCell','NetworkSubtype','NetworkType','Notes','Type','-v7.3')

NetworkProperties = zeros(NumNetworks,8);
NetworksCentCorr = zeros(17,17,NumNetworks);
NetworksCentCorrCell = cell(1,NumNetworks);
Networks_mwCMC = zeros(1,NumNetworks);
Type = zeros(1,NumNetworks);
NetworksCent = cell(1,NumNetworks);
NullNetworks_mwCMC = cell(NumNetworks,2);
NullNetProperty = cell(NumNetworks,2);

for i = 1:NumNetworks
    filename = [weighttype,'_Network_',num2str(i),'_centrality_results.mat'];  
    load(filename)
    NetworksCent{i} = Cent;
    NetworkProperties(i,:) = NetworkProps;
    NetworksCentCorr(:,:,i) = NetworkCentCorr;
    NetworksCentCorrCell{i} = NetworksCentCorr(:,:,i); 
    Networks_mwCMC(i) = Network_mwCMC;
    switch NetworkType{i}
      case 'Biological'
         Type(i) = 1;
      case 'Social'
         Type(i) = 2;
      case 'Economic'
        Type(i) = 3;
     case 'Transportation'
        Type(i) = 4;
     case 'Technological'
        Type(i) = 5;
     case 'Informational'
        Type(i) = 6;
    end 

end

%% Perform clustering

% NormCentAll is the normalised centrality scores for all measures in 
% each network (stored in a cell)
NormCentAll = cell(1,NumNetworks);
% NormCentNoRWCC is the normalised centrality scores for all measures apart 
% from random-walk closeness in each network (stored in a cell)
NormCentNoRWCC = cell(1,NumNetworks);
% NetworksLinkages is the linkages for each network
NetworksLinkages = cell(1,NumNetworks);
% NetworksCentClustDist is the distance matrix of the clusters for each 
% network
NetworksCentClustDist = cell(1,NumNetworks);
% NetworksCentClusters is a cell array where each cell is a matrix of
% clustering solutions for each network
NetworksCentClusters = cell(1,NumNetworks);
% NetworksDB is a cell array where each cell is the Davies-Bouldin indices
% for each of the identified clusters
NetworksDB = cell(1,NumNetworks);

for i = 1:NumNetworks  
    NormCentAll{i} = tiedrank(NetworksCent{i}')./size(NetworksCent{i},2);
    NormCentAll{i}(isnan(NormCentAll{i})) = 1;
    NormCentNoRWCC{i} = NormCentAll{i}(:,[1:6 8:17]);
    [NetworksLinkages{i}, NetworksCentClustDist{i}, NetworksCentClusters{i}, NetworksDB{i}] = runCentralityClustering(NormCentNoRWCC{i},50);
end

if strcmp(weighttype,'Weighted')
    
    mean_corr_weighted = nanmean(NetworksCentCorr,3);
    var_corr_weighted = nanstd(NetworksCentCorr,0,3);
    save(['Combined_',weighttype,'_Network_results.mat'],'NormCentAll','NormCentNoRWCC','mean_corr_weighted','var_corr_weighted','NetworksLinkages','NetworksCentClusters','NetworksDB','cent_names','cent_names_abbrev','Citations','NetworkNames','NetworkProperties','NetworkPropsNames','Networks','Networks_mwCMC','NetworksCent','NetworksCentCorr','NetworksCentCorrCell','NetworkSubtype','NetworkType','Notes','Type','-v7.3')

else
    
    mean_corr_unweighted = nanmean(NetworksCentCorr,3);
    var_corr_unweighted = nanstd(NetworksCentCorr,0,3);
    save(['Combined_',weighttype,'_Network_results.mat'],'NormCentAll','NormCentNoRWCC','mean_corr_unweighted','var_corr_unweighted','NetworksLinkages','NetworksCentClusters','NetworksDB','cent_names','cent_names_abbrev','Citations','NetworkNames','NetworkProperties','NetworkPropsNames','Networks','Networks_mwCMC','NetworksCent','NetworksCentCorr','NetworksCentCorrCell','NetworkSubtype','NetworkType','Notes','Type','-v7.3')

end

%% Run GLM

RunGLM

%% Run PCA

load(['Combined_',weighttype,'_Network_results.mat'])
NumNetworks = length(Networks);

for i = 1:NumNetworks  

[PCloadings{i},score{i},~,~,var_explained{i}] = pca(zscore(NetworksCent{i}([1:6 8:17],:)'));

first3PCs(i,:) = var_explained{i}(1:3);

end

save(['Combined_',weighttype,'_Network_PCA_results.mat'],'PCloadings','score','var_explained','first3PCs')

load('Combined_Unweighted_Network_results.mat','Networks_mwCMC','NetworksCent','cent_names_abbrev')

load('Weighted_Networks.mat','Corresponding_unweighted')

UnweightedCents = NetworksCent(Corresponding_unweighted);
UnweightedmwCMC = Networks_mwCMC(Corresponding_unweighted);
load('Combined_Weighted_Network_results.mat','NetworksCent','Networks_mwCMC')
WeightedCents = NetworksCent;
WeightedmwCMC = Networks_mwCMC;

Cents = cell(length(Corresponding_unweighted),1);
NetworksCentCorrCell = cell(length(Corresponding_unweighted),1);

for i = 1:length(Corresponding_unweighted)
    Cents{i} = [UnweightedCents{i}; WeightedCents{i}];
    [PCloadings{i},score{i},~,~,var_explained] = pca(zscore(Cents{i}([1:6 8:17 18:23 25:34],:)'));
    CentCorr = corr(Cents{i}','Type','Spearman'); 
    CentCorr(isnan(CentCorr)) = 0;
    NetworksCentCorrCell{i} = CentCorr;
    NetworksCentCorr(:,:,i) = CentCorr;
end

