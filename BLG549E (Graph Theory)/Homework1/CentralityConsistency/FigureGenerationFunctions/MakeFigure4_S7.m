% Make figures 4 and S7

varsbefore = who;


weighting = 'Unweighted';

% Load in results

load(['Combined_',weighting,'_Network_results.mat'])

NumNetworks = length(Networks);

% Make colourmap

Cmap = [121 85 72; 3 168 243; 74 174 78; 103 58 182; 176 176 176; 252 133 95]/255;

figure('units','pixels','outerposition',[0 0 2560 1440])
for i = 1:8
    subplot(2,4,i)

    % Plot scatter plot of network properties vs CMC

    scatter(NetworkProperties(:,i),Networks_mwCMC,50,Type,'filled','o')
    colormap(Cmap)
    xlabel(NetworkPropsNames{i})

    % Calculate the correlation

    [r, p] = corr(Networks_mwCMC',NetworkProperties(:,i),'Type','Spearman');

    % Find position for text

    xlimits = xlim;
    ylimits = ylim;
    ylim([ylimits(1) 1]);
    text(sum(xlimits)/2,.98,['\fontsize{20}\rho \fontsize{14}= ',num2str(round(r,2))],'HorizontalAlignment','center')
    caxis([1 6])

    % Only plot one ylabel for each row

    if i == 1 || i == 5
        ylabel('Mean within-network CMC');
    end
    set(gca,'FontSize',20)
end
print('Figure4.tif','-dtiff','-r300')

% Make post hoc plot of correlations between network properties and
% individual CMCs

posthoc = zeros(17,17,NumNetworks);
cmap_corrs = [make_cmap('steelblue',50,30,0);flipud(make_cmap('orangered',50,30,0))];
figure('units','pixels','outerposition',[0 0 2560 1440])
for k = 1:8
    for i = 1:16
        for j = i+1:17
            % Find the correlation between a network property and an individual
            % CMC

            T = squeeze(NetworksCentCorr(i,j,:));
            [posthoc(j,i,k),posthoc(i,j,k)] = corr(T,NetworkProperties(:,k),'Type','Spearman');
        end

    end
    subplotaxes = subplot(2,4,k);

    mat = squeeze(posthoc(:,:,k));

    [pvals,triu_ind] = triu2vec(mat,1);
 
    % Perform a Bonferroni correction

    mat(triu_ind(pvals < .05/(length(pvals)))) = 0;
    mat(triu_ind(pvals >= .05/length(pvals))) = 1;
    mat(1:length(mat)+1:end) = -1;

    P = get(subplotaxes,'pos');
    delete(subplotaxes)
    ax1 = axes('pos',P);
    
    % Make two matrices. The first is the lower triangle which contains the
    % correlations. The second is the upper triangle which indicates if the
    % correlation is significant
    
    mat_l = mat;

    mat_l(triu_ind) = NaN;
      
    mat_u = mat + mat_l';

    mat_u(triu_ind) = mat(triu_ind);
    
    % Plot the matrices one on top of the other. NaN values are set to be
    % transparent

    imagesc_clearnans(mat_l);
    colormap(ax1,cmap_corrs)    

    ax2 = axes('pos',P);
    view(2)
    imagesc_clearnans(mat_u)

    colormap(ax2,'gray')  
    set(ax2,'color','none')

    xticks(ax1,1:17)
    yticks(ax1,1:17)
    xticklabels(ax1,cent_names_abbrev)
    xtickangle(ax1,90)
    yticklabels(ax1,cent_names_abbrev)
    title(ax1,NetworkPropsNames{k})
    caxis(ax1,[-1 1])
    caxis(ax2,[-1 1])
    xticks(ax2,[])
    yticks(ax2,[])    
    set(ax1,'FontSize',20)
end
c = colorbar(ax1,'eastoutside','Fontsize',14); c.Label.String = 'Spearman correlation'; 
c.Ticks = -1:.25:1;
c.Position = [0.914115646123995,0.088962108731466,0.012885923734717,0.858464209898964];
print('FigureS7.tif','-dtiff','-r300')

% Removes variables created by this script
varsafter = who; 
varsnew = setdiff(varsafter, varsbefore); 
clear(varsnew{:})