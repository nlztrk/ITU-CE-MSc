function p = airport_network_worldplot(adj,airports,nodecolours)

% Plots a airport/airline network using the longitude and latitude of each
% airports location. Requires the mapping toolbox

% Inputs:                               adj = undirected adjacency matrix
%                                  airports = each airports IATA code
%                               nodecolours = the value to assign to each
%                                             node

% Stuart Oldham, Monash University, 2017

load airport_locations.mat

n = length(adj);
airport_location = zeros(n,2);
for i = 1:n
    Index = find(contains(globalairports,airports{i}));
    airport_location(i,:) = long_lat(Index,:);
    %airportname{i} = Airport_fullname{Index};
end
  
hold on
landareas = geoshow('landareas.shp','FaceColor',[1 1 1],'DisplayType', 'texturemap');

g = graph(adj);
p = plot(g,'Xdata',airport_location(:,2),'Ydata',airport_location(:,1));
p.NodeCData = nodecolours;
p.EdgeAlpha = .4;
p.LineWidth = .25;
p.MarkerSize = 4;

p.EdgeColor = [.5 .5 .5];

xlim([-180 180])
ylim([-90 90])
