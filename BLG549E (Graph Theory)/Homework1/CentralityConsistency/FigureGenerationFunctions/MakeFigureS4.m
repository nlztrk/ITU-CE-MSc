% Makes figure S4

load('Combined_Unweighted_Network_results.mat')
load('Combined_Unweighted_Network_PCA_results.mat')

% Selected some networks exhibiting prototypical results
Nets2Use = [151 136 96 97 23 99];

% An ordering of centrality measures I found best display the results
cent_ord = [1 3 4 7 8 10 15 2 12 13 16 5 9 11 6 14];

cent_label_ord = cent_names_abbrev([1 3 4 8 9 11 16 2 13 14 17 5 10 12 6 15]);

Cmap = [make_cmap('steelblue',50,30,0);flipud(make_cmap('orangered',50,30,0))];

% Names of example networks
example_names{1} = {'Contacts between students in a','high school in Marseilles, France (2012)'};
example_names{2} = {'Trophic-level interactions in','a freshwater stream (German)'};
example_names{3} = {'Interpersonal contacts between ','windsurfers in southern California'};
example_names{4} = {'Winnipeg road network'};
example_names{5} = {'Drug user acquaintanceships','in in Hartford, Connecticut'};
example_names{6} = {'Ahmedabad bus network'};

figure('Position',[1440 348 1496 898])

% Loop over each network, ploting the loadings of the first 3 PCs

for i = 1:6

subplot(2,3,i)

First3PCloadings = PCloadings{Nets2Use(i)}(cent_ord,1:3);

imagesc(First3PCloadings)

    explained = var_explained{Nets2Use(i)};
    XtickLabels = {['PC1 (',num2str(round(explained(1),1)),'%)'],['PC1 (',num2str(round(explained(2),1)),'%)'],['PC1 (',num2str(round(explained(3),1)),'%)']};
    yticks(1:16)
    xticks(1:3)
    xticklabels(XtickLabels)
    yticklabels(cent_label_ord)
    
    c = colorbar;
    
    colormap(Cmap)
    
    c.Label.String = 'Loading'; 
    caxis([min(abs(First3PCloadings(:))) max(abs(First3PCloadings(:)))])
    title(example_names{i})

end

print('FigureS4.tif','-dtiff','-r300')