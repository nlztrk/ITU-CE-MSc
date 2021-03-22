% Make Figure S4 and S5

varsbefore = who;

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
    CentCorr = corr(Cents{i}','Type','Spearman'); 
    CentCorr(isnan(CentCorr)) = 0;
    NetworksCentCorrCell{i} = CentCorr;
    NetworksCentCorr(:,:,i) = CentCorr;
end

figure('units','pixels','outerposition',[0 0 1920 1080])
scatter(UnweightedmwCMC,WeightedmwCMC,40,'filled')

ylabel('Weighted mean within-network CMC')
xlabel('Unweighted mean within-network CMC')
xlim([.4 1])
ylim([.4 1])
set(gca,'FontSize',20)
axis('square')
print('FigureS5.tif','-dtiff','-r300')

for i = 1:17
    cent_names_abbrev_w{i} = [cent_names_abbrev{i},'w'];
end

mean_corr_weibin = nanmean(NetworksCentCorr,3);
var_corr_weibin = nanstd(NetworksCentCorr,0,3);

cent_ind = BF_ClusterReorder(mean_corr_weibin,'corr');

cent_labels1 = [cent_names_abbrev cent_names_abbrev_w];

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

print('FigureS4.tif','-dtiff','-r300')

% Removes variables created by this script
varsafter = who; 
varsnew = setdiff(varsafter, varsbefore); 
clear(varsnew{:})