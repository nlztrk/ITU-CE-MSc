function plotClusteredData(data,clusters,varnames,colormap1,colormap2,colormap_name,xname,yname,subplotaxes)
% This function plots the results of clustering on a heatmap of the data.
% It will produce a figure where the first column shows which cluster each
% data point belongs to and the remaining columns are the variables

% Inputs:                   data = an nObjs by nVars matrix of data used to
%                                  produce the clusters. For best
%                                  visualisation, the rows should be
%                                  grouped into the clusters
%                       clusters = an array with the cluster assignment of
%                                  each data point. For best visualisation
%                                  datapoints in the same cluster should be
%                                  sorted into groups (the index of this
%                                  sorting should also be applied to the
%                                  rows of the data matrix)
%                       varnames = names of each variable (optional)
%                      colormap1 = a custom colormap for the data matrix
%                                  (optional
%                      colormap2 = a custom colormap for the clusters
%                                  (optional)
%                  colormap_name = a name for what the values represented
%                                  by colormap1 represent (optional)
%                          xname = label for the x-axis of the figure 
%                                  (optional) 
%                          yname = label for the y-axis of the figure 
%                                  (optional) 
%                    subplotaxes = axes handle of the current subplot
%                                  (optional). This is needed if you are
%                                  putting this matrix into a subplot in
%                                  order to correctly position the figure
%
% Stuart Oldham, Monash University, 2017
%
% Code adapted from https://au.mathworks.com/matlabcentral/answers/194554-how-can-i-use-and-display-two-different-colormaps-on-the-same-figure#answer_172883

nObjs = size(data,1);
nVars = size(data,2);

if nargin < 3
   varnames = num2cell(1:nVars);
end

if ~isempty(varnames)
    varnames = ['cluster' varnames]; 
end

if nargin < 4
    colormap1 = 'parula';
end

if nargin < 5
    colormap2 = 'lines';
end

if nargin < 6
    colormap_name = '';
end

if nargin < 7
    xname = '';
end

if nargin < 8
    yname = '';
end

if nargin < 9
   subplotaxes = []; 
end

if strcmp(colormap2,'auto')
    colormap2 = distinguishable_colors(max(clusters));
end

clust_mat = [clusters nan(nObjs,nVars)];
data_mat = [nan(nObjs,1) data];

%% Create the axes. 
% If the figure is being made as a subplot, the position of the subplot is
% recorded and used to position the axes. Axes can be overlayed on each
% other while subplots cannot.

if isempty(subplotaxes)
    ax1 = axes;
    imagesc_clearnans(data_mat)
    view(2)
    ax2 = axes;
    imagesc_clearnans(clust_mat)
else
    P = get(subplotaxes,'pos');
    delete(subplotaxes)
    ax1 = axes('pos',P);
    imagesc_clearnans(data_mat)
    view(2)
    ax2 = axes('pos',P);
    imagesc_clearnans(clust_mat)
end

%% Link them together
% linkaxes([ax1,ax2])

%% Set the background of the top axes to be clear and remove tickmarks
set(ax2,'color','none')
ax2.XTick = [];
ax2.YTick = [];
ax1.YTick = [];

%% Label variables
ax1.XTick = 1:nVars+1;
ax1.XTickLabels = varnames;
ax1.FontSize = 14;
xtickangle(ax1,90)
ax1.XLabel.String = xname;
ax1.YLabel.String = yname;

%% Give each one its own colormap
colormap(ax1,colormap1)
if strcmp(colormap2,'lines')
    ax2.CLim = [1 64];
end
colormap(ax2,colormap2)

%% Add in colorbar for the data matrix and resize
get(ax1,'position');

c = colorbar(ax1);
%MATLAB is slightly curious in that I need to allow time for it to add in
%the colorbbar to the figure in order to record the change in position,
%hence I pause the script for a set number of seconds
pause_time = 1;
pause(pause_time)
p = get(ax1,'position');

set(ax2,'position',p)

c.Position(3) = .25*c.Position(3);
pause(pause_time)
set(ax1,'position',p)
set(ax2,'position',p)

c.Label.String = colormap_name;
c.FontSize = 14;

end

function h = imagesc_clearnans(data)
% a wrapper for imagesc, with some formatting going on for nans

% plotting data. Removing and scaling axes (this is for image plotting)
h = imagesc(data);
%axis image off

% setting alpha values
if ismatrix(data)
  set(h, 'AlphaData', ~isnan(data))
elseif ndims(data) == 3
  set(h, 'AlphaData', ~isnan(data(:, :, 1)))
end

if nargout < 1
  clear h
end

end
