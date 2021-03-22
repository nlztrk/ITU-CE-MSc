% Make figures 4 and S7

varsbefore = who;

for W = 1:2

    if W == 1
        weighting = 'Unweighted';
        printname = 'FigureS10.tif';
    else
        weighting = 'Weighted';
        printname = 'FigureS11.tif';
    end

% Load in results

load(['Combined_',weighting,'_Network_results.mat'])
load(['Combined_',weighting,'_Network_PCA_results.mat'])

NumNetworks = length(Networks);

% Make colourmap

Cmap = [121 85 72; 3 168 243; 74 174 78; 103 58 182; 176 176 176; 252 133 95]/255;

figure('units','pixels','outerposition',[0 0 2560 1440])

for i = 1:8
    
    subplot(2,4,i)

    % Plot scatter plot of network properties vs CMC

    scatter(NetworkProperties(:,i),first3PCs(:,1),50,Type,'filled','o')
    colormap(Cmap)
    xlabel(NetworkPropsNames{i})

    % Calculate the correlation

    [r, p] = corr(first3PCs(:,1),NetworkProperties(:,i),'Type','Spearman');

    % Find position for text

    xlimits = xlim;
    ylimits = ylim;
    ylim([ylimits(1) 100]);
    text(sum(xlimits)/2,98,['\fontsize{20}\rho \fontsize{14}= ',num2str(round(r,2))],'HorizontalAlignment','center')
    caxis([1 6])

    % Only plot one ylabel for each row

    if i == 1 || i == 5
        ylabel('Variance explained by PC1');
    end
    set(gca,'FontSize',20)
end

print(printname,'-dtiff','-r300')

end

% Removes variables created by this script
varsafter = who; 
varsnew = setdiff(varsafter, varsbefore); 
clear(varsnew{:})