% Make Figure 2

varsbefore = who;

figure('units','pixels','outerposition',[0 0 2560 1440])
load('Combined_Unweighted_Network_results.mat')

unweighted_examples = [62 6 13 24 105];

for i = 1:5
    CorrVecs(:,i) = triu2vec(NetworksCentCorrCell{unweighted_examples(i)},1);
end

subplot(2,1,1)
%boxplot(CorrVecs,'Symbol','k.')
extraParams.theColors = num2cell([0 0 0;0 0 0;0 0 0;0 0 0;0 0 0],2);
JitteredParallelScatter(num2cell(CorrVecs,1),0,1,0,extraParams);

ylim([-.5 1])
xticks(1:5)
yTicks = get(gca,'ytick');
xTicks = get(gca, 'xtick');
xticks([])
minY = min(yTicks);

% You will have to adjust the offset based on the size of figure
VerticalOffset = 0.2;
HorizontalOffset = 0;

example_names{1} = {'Overlapping readership of','Slovenian periodicals'};
example_names{2} = {'Interactions in a technical','research group at a','West Virginia university'};
example_names{3} = {'Noun phrases co-occurance','in the King James Bible'};
example_names{4} = {'Friendship networks','in a Dutch school'};
example_names{5} = {'Mumbai bus network'};

for xx = 1:length(xTicks)
% Create a text box at every Tick label position
% String is specified as LaTeX string and other appropriate properties are set
    text(xTicks(xx) - HorizontalOffset, minY - VerticalOffset, example_names{xx}, 'FontName','Helvetica','HorizontalAlignment','center','FontSize',20)
end

ylabel({'Centrality Measure';'Correlations'})
title('Unweighted networks')
set(gca,'FontSize',20)
  
subplot(2,1,2)

load('Combined_Weighted_Network_results.mat')

weighted_examples = [16 18 12 25 38];

for i = 1:5
CorrVecs(:,i) = triu2vec(NetworksCentCorrCell{weighted_examples(i)},1);
end

%boxplot(CorrVecs,'Symbol','k.')
extraParams.theColors = num2cell([0 0 0;0 0 0;0 0 0;0 0 0;0 0 0],2);
JitteredParallelScatter(num2cell(CorrVecs,1),0,1,0,extraParams);
ylim([-.5 1])
xticks(1:5)
yTicks = get(gca,'ytick');
xTicks = get(gca, 'xtick');
xticks([])
minY = min(yTicks);
% You will have to adjust the offset based on the size of figure
VerticalOffset = 0.2;
HorizontalOffset = 0;

example_names{1} = {'Overlapping readership of','Slovenian periodicals'};
example_names{2} = {'Karate club'};
example_names{3} = {'Co-occurance of food ingredients','in a database of recipes'};
example_names{4} = {'Star Wars Episode II','character co-apperances'};
example_names{5} = {'Kolkata bus network'};

for xx = 1:length(xTicks)
% Create a text box at every Tick label position
% String is specified as LaTeX string and other appropriate properties are set
text(xTicks(xx) - HorizontalOffset, minY - VerticalOffset, example_names{xx}, 'FontName','Helvetica','HorizontalAlignment','center','FontSize',20)
end

ylabel({'Centrality Measure';'Correlations'})
title('Weighted networks')
set(gca,'FontSize',20)
  
%print('Figure2.png','-dpng','-r300')
print('Figure2.tif','-dtiff','-r300')

% Removes variables created by this script

varsafter = who; 
varsnew = setdiff(varsafter, varsbefore); 
clear(varsnew{:})