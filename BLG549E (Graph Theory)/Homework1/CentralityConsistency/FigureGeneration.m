% This script will generate all the figures present in the paper that were
% generated based are the data obtained. Other figures were created
% manually in powerpoint using an excessive amount of autoshapes

% This script also uses colormaps from cbrewer

%% Initial setup
% Define these variables for your own environment and desired parameters
% Define path to the directory of this script
MAINPATH = '/scratch/kg98/stuarto/CentralityConsistency-master';
addpath(genpath(MAINPATH))

%% Figure 2

MakeFigure2

%% Figure 3

MakeFigure3

%% Figure 4 and S7

MakeFigure4_S7

%% Figure 5 and 6

MakeFigure5_6

%% Figure 7-8, S12-S17

% Note when making the figure of the airport network it will flash up a
% warning. Figure will produce just fine however.

MakeClusterFigures('Unweighted',72,8)
print('Figure7.tif','-dtiff','-r300')

MakeClusterFigures('Unweighted',127,3)
print('Figure8.tif','-dtiff','-r300')

MakeClusterFigures('Unweighted',12,6)
print('FigureS15.tif','-dtiff','-r300')

MakeClusterFigures('Unweighted',23,9)
print('FigureS16.tif','-dtiff','-r300')

MakeClusterFigures('Unweighted',206,7)
print('FigureS17.tif','-dtiff','-r300')

MakeClusterFigures('Unweighted',13,3)
print('FigureS18.tif','-dtiff','-r300')

MakeClusterFigures('Unweighted',20,3)
print('FigureS19.tif','-dtiff','-r300')

MakeClusterFigures('Unweighted',52,3)
print('FigureS20.tif','-dtiff','-r300')

%% Figure S2

MakeFigureS2

%% Figure S3

MakeFigureS3

%% Figure S4

MakeFigureS4

%% Figure S5 and S6

MakeFigureS5_S6

%% Figure S7 and S9

MakeFigureS7_S9

%% Figure S10 and S11

MakeFigureS10_S11

%% Figure S12

MakeFigureS12

%% Figure S13 and S14

MakeFigureS13_S14