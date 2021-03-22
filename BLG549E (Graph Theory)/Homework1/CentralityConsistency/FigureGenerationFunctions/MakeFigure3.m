% makes figure 3

varsbefore = who;

load('Combined_Weighted_Network_results.mat','mean_corr_weighted','var_corr_weighted')
load('Combined_Unweighted_Network_results.mat','mean_corr_unweighted','var_corr_unweighted','cent_names_abbrev')

% Order unweighted centrality measures by their similarity. Use this
% ordering to order centrality measures in plots

cent_ind = BF_ClusterReorder(mean_corr_unweighted,'corr');

%figure('units','pixels','outerposition',[0 0 1080 1080])

figure('units','pixels','outerposition',[0 0 2250 1080])

% Make colour maps

nice_cmap = [make_cmap('steelblue',50,30,0);flipud(make_cmap('orangered',50,30,0))];

positive_cmap = flipud(make_cmap('orangered',50,30,0));

subplot(1,4,1)

data = mean_corr_unweighted;

% order centrality measures

cent_labels = cent_names_abbrev(cent_ind);
Num_cents = length(cent_names_abbrev);

% Plot matrix of mean between-network CMCs in unweighted networks

imagesc(data(cent_ind,cent_ind))
axis square
colormap(gca,nice_cmap)
caxis([-1 1])
c = colorbar(gca,'Fontsize',10); c.Label.String = 'Spearman correlation'; 
fig_name = {'Mean Spearman correlation';'across unweighted networks'};
title(fig_name,'interpreter','none','Fontsize',14)
xticks(1:Num_cents)
yticks(1:Num_cents)
xticklabels(cent_labels)
xtickangle(90)
yticklabels(cent_labels)

subplot(1,4,2)

data = var_corr_unweighted;

% Plot matrix of the standard deviations of correlations across unweighted
% networks

imagesc(data(cent_ind,cent_ind))
axis square
colormap(gca,positive_cmap)
%caxis([min(min(data)) max(max(data))])
caxis([0 .5])
c = colorbar(gca,'Fontsize',10); c.Label.String = 'Standard deviation'; 
fig_name = {'Spearman correlation standard';'deviation across unweighted networks'};
title(fig_name,'interpreter','none','Fontsize',14)
xticks(1:Num_cents)
yticks(1:Num_cents)
xticklabels(cent_labels)
xtickangle(90)
yticklabels(cent_labels)  

subplot(1,4,3)
  
% Plot matrix of mean between-network CMCs in weighted networks

data = mean_corr_weighted;

cent_labels = cent_names_abbrev(cent_ind);

imagesc(data(cent_ind,cent_ind))
axis square  

colormap(gca,nice_cmap)
caxis([-1 1])
c = colorbar(gca,'Fontsize',10); c.Label.String = 'Spearman correlation'; 
fig_name = {'Mean Spearman correlation';'across weighted networks'};
title(fig_name,'interpreter','none','Fontsize',14)
xticks(1:Num_cents)
yticks(1:Num_cents)
xticklabels(cent_labels)
xtickangle(90)
yticklabels(cent_labels)

subplot(1,4,4)

% Plot matrix of the standard deviations of correlations across weighted
% networks

data = var_corr_weighted;

imagesc(data(cent_ind,cent_ind))
axis square
colormap(gca,positive_cmap)
%caxis([min(min(data)) max(max(data))])
caxis([0 .5])
c = colorbar(gca,'Fontsize',10); c.Label.String = 'Standard deviation'; 
fig_name = {'Spearman correlation standard';'deviation across weighted networks'};
title(fig_name,'interpreter','none','Fontsize',14)
xticks(1:Num_cents)
yticks(1:Num_cents)
xticklabels(cent_labels)
xtickangle(90)
yticklabels(cent_labels)

print('Figure3.tif','-dtiff','-r300')

% Removes variables created by this script
varsafter = who; 
varsnew = setdiff(varsafter, varsbefore); 
clear(varsnew{:})