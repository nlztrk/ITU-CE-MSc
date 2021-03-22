% Make Figure S3

varsbefore = who;

load('Combined_Unweighted_Network_results.mat')

% Get the median CMC to order the CMC distributions (looks visually nice
% than ordering by the mean)

Networks_medianCMC = zeros(length(Networks),1);
for i = 1:212
Networks_medianCMC(i) = median(triu2vec(NetworksCentCorrCell{i},1));
end

% Sort networks from high to low median CMC

[~,sortedCMCinds] = sort(Networks_medianCMC,'descend');


Cmap = [121 85 72; 3 168 243; 74 174 78; 103 58 182; 176 176 176; 252 133 95]/255;

% Get all the CMCs for each unweighted network

for i = 1:212
    CorrVecs(:,i) = triu2vec(NetworksCentCorrCell{sortedCMCinds(i)},1);
end

figure('units','pixels','outerposition',[0 0 2560 1440])

% Plot each boxplot and record the location. Then plot a patch with a
% colour signifying the networks type

% Plot the first half of the unweighted networks

s1 = subplot(3,1,1);

boxplot(CorrVecs(:,1:106),'Symbol','k.') 

h = findobj(gca,'Tag','Box');

for j=1:length(h)
    patch(get(h(107-j),'XData'),get(h(107-j),'YData'),Cmap(Type(sortedCMCinds(j)),:));
end

hold on 

boxplot(CorrVecs(:,1:106),'Colors','k','Symbol','k.') 

set(gca,'xtick',[])

xtickangle(90)

title('Unweighted')

y_min(1) = min(ylim);

ylabel({'Centrality Measure';'Correlations'})

xlimits = xlim;

xlim([xlimits(1)-1 xlimits(2)+1])

set(gca,'FontSize',16)

% Plot the second half of the unweighted networks

s2 = subplot(3,1,2);

boxplot(CorrVecs(:,107:212),'Symbol','k.') 

h = findobj(gca,'Tag','Box');

for j=1:length(h)
    patch(get(h(107-j),'XData'),get(h(107-j),'YData'),Cmap(Type(sortedCMCinds(j+106)),:));
end

hold on 

boxplot(CorrVecs(:,107:212),'Colors','k','Symbol','k.') 

set(gca,'xtick',[])

xtickangle(90)

title('Unweighted continued')

y_min(2) = min(ylim);

ylabel({'Centrality Measure';'Correlations'})

xlimits = xlim;

xlim([xlimits(1)-1 xlimits(2)+1])

set(gca,'FontSize',16)

s3 = subplot(3,1,3);

load('Combined_Weighted_Network_results.mat')

numNetworks = length(Networks);

clear CorrVecs CorrVecsCell

Networks_medianCMC = zeros(length(Networks),1);

for i = 1:numNetworks
    Networks_medianCMC(i) = median(triu2vec(NetworksCentCorrCell{i},1));
end

[~,sortedCMCinds] = sort(Networks_medianCMC,'descend');

Cmap = [121 85 72; 3 168 243; 74 174 78; 103 58 182; 176 176 176; 252 133 95]/255;

% Plot each boxplot and record the location. Then plot a patch with a
% colour signifying the networks type

for i = 1:numNetworks
    CorrVecs(:,i) = triu2vec(NetworksCentCorrCell{sortedCMCinds(i)},1);
end

boxplot(CorrVecs,'Symbol','k.') 

h = findobj(gca,'Tag','Box');

for j=1:length(h)
    patch(get(h(numNetworks+1-j),'XData'),get(h(numNetworks+1-j),'YData'),Cmap(Type(sortedCMCinds(j)),:));
end

hold on 

boxplot(CorrVecs,'Colors','k','Symbol','k.')  

set(gca,'xtick',[])

xtickangle(90)

title('Weighted')

y_min(3) = min(ylim);

ylabel({'Centrality Measure';'Correlations'})

yrange = [-1 1];

set(s1,'ylim',yrange)

set(s2,'ylim',yrange)

set(s3,'ylim',yrange)

xlimits = xlim;

xlim([xlimits(1)-.5 xlimits(2)+.5])

set(gca,'FontSize',16)

print('FigureS3.tif','-dtiff','-r300')

% Removes variables created by this script
varsafter = who; 
varsnew = setdiff(varsafter, varsbefore); 
clear(varsnew{:})