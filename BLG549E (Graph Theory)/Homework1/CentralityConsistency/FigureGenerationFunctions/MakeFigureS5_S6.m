% Make Figure S5 and S6

varsbefore = who;

load('Combined_Unweighted_Network_results.mat','Networks_mwCMC','NetworksCent','cent_names_abbrev')

% Load in the index of unweighted networks that have a corresponding
% weighted network
load('Weighted_Networks.mat','Corresponding_unweighted')

NumNetworks = length(Corresponding_unweighted);

% Store unweighted data
UnweightedCents = NetworksCent(Corresponding_unweighted);
UnweightedmwCMC = Networks_mwCMC(Corresponding_unweighted);

load('Combined_Weighted_Network_results.mat','NetworksCent','Networks_mwCMC')

% Store weighted data
WeightedCents = NetworksCent;
WeightedmwCMC = Networks_mwCMC;

Cents = cell(length(Corresponding_unweighted),1);

NetworksCentCorrCell = cell(length(Corresponding_unweighted),1);

% Correlate weighted and unweighted centrality measures

NetworksCentCorr = zeros(NumNetworks,NumNetworks,3);

for i = 1:NumNetworks
    Cents{i} = [UnweightedCents{i}; WeightedCents{i}];
    CentCorr = corr(Cents{i}','Type','Spearman'); 
    CentCorr(isnan(CentCorr)) = 0;
    NetworksCentCorrCell{i} = CentCorr;
    NetworksCentCorr(:,:,i) = CentCorr;
end

% Plot scatter plot of unweighted vs weighted mean-within centrality
% measure correlations (Figure S6)

figure('units','pixels','outerposition',[0 0 1920 1080])
scatter(UnweightedmwCMC,WeightedmwCMC,40,'filled')

ylabel('Weighted mean within-network CMC')
xlabel('Unweighted mean within-network CMC')
xlim([.4 1])
ylim([.4 1])
set(gca,'FontSize',20)
axis('square')
print('FigureS6.tif','-dtiff','-r300')

% Plot matrix of mean correlation and std of correlations for each pair of
% centrality measures (Figure S5)

% Create labels for unweighted and weighted centrality measures

cent_names_abbrev_w = cell(1,cent_names_abbrev);

for i = 1:length(cent_names_abbrev_w)
    cent_names_abbrev_w{i} = [cent_names_abbrev{i},'w'];
end

cent_labels1 = [cent_names_abbrev cent_names_abbrev_w];

% Get mean and std of correlations between all (unweighted and weighted)
% centrality measures

mean_corr_weibin = nanmean(NetworksCentCorr,3);
var_corr_weibin = nanstd(NetworksCentCorr,0,3);

% Find ordering

cent_ind = BF_ClusterReorder(mean_corr_weibin,'corr');

figure('units','pixels','outerposition',[0 0 1920 1080])

nice_cmap = [make_cmap('steelblue',50,30,0);flipud(make_cmap('orangered',50,30,0))];

positive_cmap = flipud(make_cmap('orangered',50,30,0));

subplot(1,2,1)

data = mean_corr_weibin;

cent_labels = cent_labels1(cent_ind);
Num_cents = length(cent_labels);
imagesc(data(cent_ind,cent_ind))
axis square
colormap(gca,nice_cmap)
caxis([-1 1])
c = colorbar(gca,'Fontsize',14); c.Label.String = 'Spearman correlation'; 

fig_name = {'Mean Spearman correlation'};
title(fig_name,'interpreter','none','Fontsize',16)
xticks(1:Num_cents)
yticks(1:Num_cents)
xticklabels(cent_labels)
xtickangle(90)
yticklabels(cent_labels)

subplot(1,2,2)

data = var_corr_weibin;

imagesc(data(cent_ind,cent_ind))
axis square
colormap(gca,positive_cmap)
caxis([0 .55])
c = colorbar(gca,'Fontsize',14); c.Label.String = 'Standard deviation'; 
fig_name = {'Spearman correlation standard deviation'};
title(fig_name,'interpreter','none','Fontsize',16)
xticks(1:Num_cents)
yticks(1:Num_cents)
xticklabels(cent_labels)
xtickangle(90)
yticklabels(cent_labels)

print('FigureS5.tif','-dtiff','-r300')

% Removes variables created by this script
varsafter = who; 
varsnew = setdiff(varsafter, varsbefore); 
clear(varsnew{:})