% Make figure S12

varsbefore = who;

figure('units','pixels','outerposition',[0 0 1920 1080])

subplot(1,2,1)

load('Combined_Unweighted_Network_results.mat')

% Find correlations between properties in unweighted networks

unweighted_prop_corrs = corr(NetworkProperties,'Type','Spearman');
imagesc(unweighted_prop_corrs)
xticks(1:8)
yticklabels(NetworkPropsNames)
xticklabels(NetworkPropsNames)
xtickangle(45)
title('Unweighted networks')
axis square
caxis([-1 1])
set(gca,'FontSize',20)
subplot(1,2,2)

load('Combined_Weighted_Network_results.mat')

% Find correlations between properties in weighted networks

weighted_prop_corrs = corr(NetworkProperties,'Type','Spearman');

imagesc(weighted_prop_corrs)
xticks(1:8)
yticklabels(NetworkPropsNames)
xticklabels(NetworkPropsNames)
xtickangle(45)
title('Weighted networks')
axis square
cmap = [make_cmap('steelblue',50,30,0);flipud(make_cmap('orangered',50,30,0))];
caxis([-1 1])
set(gca,'FontSize',20)
colormap(cmap)
c = colorbar('eastoutside','Fontsize',14); 
c.Label.String = 'Spearman correlation'; 
c.Position = [0.9231    0.1620    0.0201    0.6791];

print('FigureS12.tif','-dtiff','-r300')

% Removes variables created by this script
varsafter = who; 
varsnew = setdiff(varsafter, varsbefore); 
clear(varsnew{:})