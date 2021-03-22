function MakeClusterFigures(Weighttype,NetworkNum,usecluster)

MAINPATH = '/scratch/kg98/stuarto/CentralityConsistency-master';
% Define path to the directory of the BCT
BCTPATH = '/projects/kg98/stuarto/BCT';
addpath(genpath(MAINPATH))
addpath(genpath(BCTPATH))


load(['Combined_',Weighttype,'_Network_results.mat'],'Networks','NormCentAll','NormCentNoRWCC','NetworksLinkages','NetworksCentClusters','NetworksDB','cent_names_abbrev')


    % Initialize variables
    
    dataMatrix = NormCentAll{NetworkNum};
    dataMatrix2 = NormCentNoRWCC{NetworkNum};
    Z = NetworksLinkages{NetworkNum}; 
    %D = NetworksCentClustDist{NetworkNum};
    Y = pdist(dataMatrix2,'euclidean');
    D = squareform(Y);
    Hclusters = NetworksCentClusters{NetworkNum};
    DB = NetworksDB{NetworkNum};
    centrality_labels = cent_names_abbrev;

    figure('units','pixels','outerposition',[0 0 2560 1440])
    %figure('units','pixels','outerposition',[0 0 1600 900])

    % Get optimal node order and select desired cluster
    order = optimalleaforder(Z,D);
    chosen_clusters = Hclusters(:,usecluster);

    % Order clusters by their average score
    cluster_mean = zeros(usecluster,1);

    for j = 1:usecluster
        cluster_mean(j) = mean(mean(dataMatrix2(chosen_clusters==j,:)));
    end

    [~,mean_cluster_order] = sort(cluster_mean,'descend');
    
    % Once the order of clusters is found, find the node orderings within
    % that cluster

    obj_cluster = zeros(size(chosen_clusters));
    
    for j = 1:usecluster
        obj_cluster(chosen_clusters==mean_cluster_order(j)) = j;
    end

    [cluster_ord,obj_ord] = sort(obj_cluster,'ascend');

    % Plot DB values
    subplot(6,6,[3 4 9 10])

    h(1) = scatter(usecluster,DB(usecluster),'r','filled');
    hold on
    h(2) = plot(DB,'-o','MarkerFaceColor',lines(1),'Color',lines(1));
    ylabel('DB values','Fontsize',14)
    xlabel('Clusters','Fontsize',14)
    uistack(h(1),'top');

    % Plot clustergram
    
    ax = subplot(6,6,[1 2 7 8]);

    clustergram_distances(D,Z,[2 usecluster],'',[make_cmap('orangered',50,30,0);flipud(make_cmap('steelblue',50,30,0))],'Euclidean distance',ax);

    if usecluster < 8
    cmap2 = cbrewer('qual','Set1',usecluster);
    else
        cmap2 = [cbrewer('qual','Set1',9); 0 0.5 0];
    end
    
    % Get optimalordering of variables (centrality measures)

    var_order = BF_ClusterReorder(dataMatrix');
    
    % Plot clustered data

    ax2 = subplot(6,6,[13 14 19 20 25 26 31 32]);

    plotClusteredData(dataMatrix(obj_ord,var_order),cluster_ord,centrality_labels(var_order),[make_cmap('steelblue',50,30,0);flipud(make_cmap('orangered',50,30,0))],cmap2,'Normalised Centrality','','',ax2);

    % Plot clustered data with just two clusters
    
    ax3 = subplot(6,6,[5 6 11 12]);

    chosen_clusters2 = Hclusters(:,2);

    obj_ord2 = obj_ord;
    cluster_ord2 = chosen_clusters2(obj_ord);

    plotClusteredData(dataMatrix(obj_ord2,var_order),cluster_ord2,[],[make_cmap('steelblue',50,30,0);flipud(make_cmap('orangered',50,30,0))],[0 0 0; .5 .5 .5],'Normalised Centrality','','',ax3);

    % Plot the network, colouring nodes by cluster assignment
    
    subplot(6,6,[15:18 21:24 27:30 34:36])

        G = graph(Networks{NetworkNum});
        g = plot(G,'Layout','force');    
        g.NodeCData = obj_cluster;
        g.EdgeAlpha = .5;
        g.LineWidth = .5;
        g.MarkerSize = 6;
        g.EdgeColor = [.5 .5 .5];
        g.NodeLabel = {};
        colormap(gca,cmap2);
        XData = g.XData;
        YData = g.YData;
        xlim([min(XData) max(XData)])
        ylim([min(YData) max(YData)])
        axis off

