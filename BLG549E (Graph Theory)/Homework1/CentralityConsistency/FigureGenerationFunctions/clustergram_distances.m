function [T, order] = clustergram_distances(D,Z,thresh,varnames,colormap1,colormap_name,subplotaxes)
%
% This function plots a dendrogram next to its distance matrix
%
% Inputs:                      D = distance matrix
%                              Z = linkages
%                         thresh = threshold to use for cutting dendrogram
%                       varnames = names of each variable (optional)
%                      colormap1 = a custom colormap for the data matrix
%                                  (optional
%                  colormap_name = a name for what the values represented
%                                  by colormap1 represent (optional)
%                    subplotaxes = axes handle of the current subplot
%                                  (optional). This is needed if you are
%                                  putting this matrix into a subplot in
%                                  order to correctly position the figure
%
% Outputs:                     T = a matrix of cluster membership
%                          order = optimal object order
%
% Stuart Oldham, Monash University, 2017

% Code adapted from https://au.mathworks.com/matlabcentral/answers/194554-how-can-i-use-and-display-two-different-colormaps-on-the-same-figure#answer_172883

nObjs = size(D,1);
nVars = size(D,2);

if nargin < 3
    thresh = [];
end

if nargin < 4
   varnames = num2cell(1:nVars);
end

if isempty(varnames)
    varnames = cell(1,nVars);
    noxticks = 1;
else
    noxticks = 0;
end

if nargin < 5
    colormap1 = 'parula';
end

if nargin < 6
    colormap_name = '';
end

if nargin < 7
    subplotaxes = [];
end

order = optimalleaforder(Z,D);

varnames = ['cluster' varnames(flip(order))];

%% Find the area available for plotting the distance matrix

total_area = ceil(nVars/.8);
    dend_area = total_area - nVars;

    data = D(flip(order),flip(order));
    
    data_mat = [nan(nObjs,dend_area) data];
    
%% If plotting on a subplot extract that info    
    
if isempty(subplotaxes)
    ax1 = axes;
else
    P = get(subplotaxes,'pos');
    delete(subplotaxes)
    ax1 = axes('pos',P);
end

%% Plot the distance matrix

imagesc_clearnans(data_mat)

%% Perform clustering and put a rectange around the clusters (put 
% rectangles around the two cluster solution and the selected solution)

if ~isempty(thresh)
    thresh_sort = sort(thresh,'ascend');
    for i = 1:length(thresh)
        T(:,i) = cluster(Z,'MaxClust',thresh_sort(i));
    end
    
    rectangle_colors = [0 0 0; .5 .5 .5];

    for j = 1:length(thresh_sort)
        T_ord = flip(T(order,j));
        for i = 1:thresh_sort(j)
           ind = find(T_ord == i); 
           rectangle('Position',[min(ind)+dend_area-.5 min(ind)-.5 length(ind) length(ind)],'EdgeColor',rectangle_colors(j,:),...
            'LineWidth',length(thresh_sort)+2-j)
        end
    end
end

axis off

view(2)

if isempty(subplotaxes)
    ax2 = axes;
else
    ax2 = axes('pos',P);
end

%% Plot the dendrogram and align it

    dendrogram(Z,0,'reorder',order,'Orientation','Left')
    
    xlims = xlim;
    ylims = ylim;
    ylim([.5 ylims(2)-.5])
    xlim_range = xlims(2) - xlims(1);

    xlim([(0 - xlim_range/1000000 - xlim_range*4)  xlims(2)])

axis off
%% Set the background of the top axes to be clear and remove tickmarks
set(ax2,'color','none')
ax2.XTick = [];
ax2.YTick = [];
ax1.YTick = [];
%% Label variables
if noxticks
    ax1.XTick = [];
else
    ax1.XTick = 5:nVars+5;
    ax1.XTickLabels = varnames;
    set(ax1,'Fontsize',14)
    ax1.XTickLabelRotation = 90;
end

%% Give each one its own colormap
colormap(ax1,colormap1)
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
if ndims(data) == 2
  set(h, 'AlphaData', ~isnan(data))
elseif ndims(data) == 3
  set(h, 'AlphaData', ~isnan(data(:, :, 1)))
end

if nargout < 1
  clear h
end

end
