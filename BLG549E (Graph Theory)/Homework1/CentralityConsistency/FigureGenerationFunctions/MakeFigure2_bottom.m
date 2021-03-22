% Makes the bottom half of figure 2.

varsbefore = who;

nice_cmap = [make_cmap('steelblue',50,30,0);flipud(make_cmap('orangered',50,30,0))];
figure('units','pixels','outerposition',[0 0 1920 1080])

cent_labels = cent_names_abbrev;
cent_labels = cent_labels(cent_ind);

selected_unweighted = [8 5 7];
selected_weighted = [12 15 11];
caxis_range = [-.3 1];

for j = 1:length(selected_unweighted)
    subplot(9,3,(0:3:6)+j)
    i = selected_unweighted(j);
    data = corr(NetworksCent{i}','Type','Spearman');
    imagesc(data(cent_ind,cent_ind))
    axis('square')
    colormap(gca,nice_cmap)
    caxis(caxis_range)
    fig_name = sprintf('%s',net_fullName{i});
    xticks(1:15)
    yticks(1:15)
    xticklabels(cent_labels)
    xtickangle(90)
    yticklabels(cent_labels)
    set(gca, 'FontSize', 14)
    title(fig_name,'interpreter','none','Fontsize',18)
end

for j = 1:length(selected_weighted)
    subplot(9,3,(12:3:18)+j)
    i = selected_weighted(j);
    data = corr(NetworksCent{i}','Type','Spearman');
    imagesc(data(cent_ind,cent_ind))
    axis('square')
    colormap(gca,nice_cmap)
    caxis(caxis_range)
    fig_name = sprintf('%s',net_fullName{i});
    xticks(1:15)
    yticks(1:15)
    xticklabels(cent_labels)
    xtickangle(90)
    yticklabels(cent_labels)
    set(gca, 'FontSize', 14)
    title(fig_name,'interpreter','none','Fontsize',18)
end

c = colorbar('southoutside','Fontsize',14); c.Label.String = 'Spearman correlation'; 
c.Ticks = caxis_range(1):.1:caxis_range(2);
c.Position = [0.1450 0.1601 0.7295 0.0304];

% Removes variables created by this script
varsafter = who; 
varsnew = setdiff(varsafter, varsbefore); 
clear(varsnew{:})