% Makes the top half of figure 2

varsbefore = who;

cents_data_corr = NetworksCentCorrCell;

cents_data_corr{16} = [];

% For this script rename the abbreviations for the weighted networks

net_abbrevname_temp = net_abbrevname;

net_abbrevname_temp{11} = 'Netsci';
net_abbrevname_temp{12} = 'LesMis';
net_abbrevname_temp{13} = 'Journal';
net_abbrevname_temp{15} = 'Brain';

% To produce the plot we need a blank space

net_abbrevname_temp{16} = '';

% Get the vector of correlation values and find the mean and minimum

corr_vec = cell(1,16);
net_mean_corr = zeros(1,15);
minval = zeros(1,15);

for i = 1:15  
     C = triu2vec(cents_data_corr{i},1);
     corr_vec{i} = C;
     net_mean_corr(i) = mean(C);
     minval(i) = min(C);
end

corr_vec{16} = 100;

% Find the order of unweighted and weighted networks. Ordered by mean
% correlation

[~,corr_ord_bin] = sort(net_mean_corr(unweighted_network_inds),'descend');
[~,corr_ord_wei] = sort(net_mean_corr(weighted_network_inds),'descend');

ord = [unweighted_network_inds(corr_ord_bin) 16 weighted_network_inds(corr_ord_wei)]; 

netcolor_all = [netcolor [0 0 0]];

netcolor_all_ord = netcolor_all(ord);

figure('units','pixels','outerposition',[0 0 1920 1080])

% To make the plot work two axis need to be overlayed

% Subplot is used to get the desired location of the second axis
ax2 = subplot(4,16,[12:16 28:32 44:48]);
location = ax2.Position;

ax1 = subplot(4,16,[1:12 17:28 33:44]);

% Unweighted networks are plotted

corr_vec_ord = corr_vec(ord);
name_ord = net_abbrevname_temp(ord);
extraParams.theColors = netcolor_all_ord(1:12);
JitteredParallelScatter(corr_vec_ord(1:12),0,1,0,extraParams);
ylim([-.3 1])
xticks(1:11)
xlim([.5 13])

xticklabels(name_ord(1:11))
xtickangle(45)
set(gca,'fontsize',16);
ylabel('Spearman coefficient','Fontsize',20)
title('Unweighted networks','Fontsize',20)

% Put the second axis in the desired location

axes('Position', location);

% Weighted networks are plotted

extraParams.theColors = netcolor_all_ord(12:16);
JitteredParallelScatter(corr_vec_ord(12:16),0,1,0,extraParams);
xlim([1 5.5])
ylim([-.3 1])
xticks(2:5)
xticklabels(13:16)
xtickangle(45)
set(gca,'fontsize',16);
xticklabels(name_ord(13:16))
yticks([])
title('Weighted networks','Fontsize',20)
plot([1 1],[-.29 .99],'w--','LineWidth',2)

% Removes variables created by this script
varsafter = who; 
varsnew = setdiff(varsafter, varsbefore); 
clear(varsnew{:})