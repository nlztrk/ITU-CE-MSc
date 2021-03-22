% makes Figure S2

varsbefore = who;

figure('units','pixels','outerposition',[0 0 1920 1080])

% Creates network with discordant centrality

A = [0     1     1     0     0     0     0     0     0;...
     1     0     0     0     0     0     0     1     1;...
     1     0     0     0     0     1     1     0     0;...
     0     0     0     0     1     0     0     1     1;...
     0     0     0     1     0     0     1     0     1;...
     0     0     1     0     0     0     1     1     0;...
     0     0     1     0     1     1     0     0     0;...
     0     1     0     1     0     1     0     0     0;...
     0     1     0     1     1     0     0     0     0];
 
% Rearrange the node order because I messed it up 
 
node_ord = [1 4 3 9 6 2 7 5 8];

A = A(node_ord,node_ord);

% Calculate centrality measures and find the rank

A_cent(:,1) = sum(A);
A_cent(:,2) = closeness_bin(A);
A_cent(:,3) = eigenvector_centrality_und(A);
A_cent(:,4) = betweenness_bin(A);
A_ties = tiedrank(A_cent);

% Get the appropriate colormap

cmap = flipud(cbrewer('div','RdYlBu', 9));

% Plot the network and get node coordinates

subplot(2,3,1)

g = plot(graph(A),'Layout','force');    
g.NodeCData = 1:9;
g.EdgeAlpha = 1;
g.LineWidth = 3;
g.MarkerSize = 20;
g.EdgeColor = [0 0 0];
g.NodeLabel = {};
colormap(cmap);
Xcoords = g.XData;
Ycoords = g.YData;

axis off

% Plot the networks domination relationships (also uses the existing node
% coordinates

subplot(2,3,2)

g = plot(graph(A),'Layout','force');    
g.NodeCData = 1:9;
g.EdgeAlpha = 0;
g.LineWidth = 3;
g.MarkerSize = 20;
g.EdgeColor = [0 0 0];
g.NodeLabel = {};
colormap(cmap);
g.XData = Xcoords;
g.YData = Ycoords;

axis off

% Plot the centrality ranks

subplot(2,3,3)

for i = 1:9
plot(A_ties(i,:),'color',cmap(i,:),'LineWidth',5)
hold on
end
xticks(1:4);
xticklabels({'DC','CC','EC','BC'});
ax1 = gca;
pause(1)
yruler = ax1.YRuler;
yruler.Axle.Visible = 'off';
xruler = ax1.XRuler;
xruler.Axle.Visible = 'off';

yticklabels(flip(1:9))
box off
ax1.FontSize = 14;
ylabel('Centrality rank')

cmap = flipud(cbrewer('div','RdYlBu', 9));

% Create the network of concordant centrality

B = zeros(9);
for i = 1:9
    B(i,10-i:9) = 1;
end
B(1:10:end)=0;

% Calculate centrality
B_cent(:,1) = sum(B);
B_cent(:,2) = closeness_bin(B);
B_cent(:,3) = eigenvector_centrality_und(B);
B_cent(:,4) = betweenness_bin(B);
B_ties = tiedrank(B_cent);

% Create the dominance relations

B_dir = zeros(9);

for i = 1:9
   B_dir(i,i+1:9) = 1; 
end
B_dir(5,4) = 1;

subplot(2,3,5)

% Plot the network then manually adjust

dg = plot(digraph(B_dir),'Layout','Force');
dg.NodeCData = [1 2 3 4 4 5 6 7 8];
dg.EdgeAlpha = 1;
dg.LineWidth = 1;
dg.ArrowSize = 15;
dg.MarkerSize = 20;
dg.EdgeColor = [0 0 0];
dg.NodeLabel = {};
colormap(cmap);
Xcoords = dg.XData;
Ycoords = dg.YData;
Xcoords(1) = -0.07;
Ycoords(1) = .52;
Xcoords(3) = -0.75;
Ycoords(3) = 1.7;
Xcoords(5) = 0.5;
Xcoords(6) = -1.75;
Ycoords(6) = -1.25;
Ycoords(7) = .5;
Ycoords(9) = 1;
Xcoords(9) = -1.75;
dg.XData = Xcoords;
dg.YData = Ycoords;
colormap(cmap);

axis off

subplot(2,3,4)

g = plot(graph(B),'Layout','force');    
g.NodeCData = [1 2 3 4 4 5 6 7 8];
g.EdgeAlpha = 1;
g.LineWidth = 3;
g.MarkerSize = 20;
g.EdgeColor = [0 0 0];
g.NodeLabel = {};
colormap(cmap);
g.XData = Xcoords;
g.YData = Ycoords;

axis off
% Plot the centrality ranks

subplot(2,3,6);

INT = 1;
for i = 1:9
plot(B_ties(i,:),'color',cmap(INT,:),'LineWidth',5)
hold on
if i ~= 4
INT = INT + 1;
end
if i == 5
INT = INT + 1;
end
end
xticks(1:4);
xticklabels({'DC','CC','EC','BC'});
ax1 = gca;
pause(1)
yruler = ax1.YRuler;
yruler.Axle.Visible = 'off';
xruler = ax1.XRuler;
xruler.Axle.Visible = 'off';

yticklabels(flip(1:9))
box off
ax1.FontSize = 14;
ylabel('Centrality rank')

% Removes variables created by this script
varsafter = who; 
varsnew = setdiff(varsafter, varsbefore); 
clear(varsnew{:})