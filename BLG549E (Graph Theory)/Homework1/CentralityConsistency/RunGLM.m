load('Combined_Unweighted_Network_results.mat')

idx = [1 2 5 6 7 8];
[b,dev,stats] = glmfit(NetworkProperties(:,idx),Networks_mwCMC');
r_squared = sqrt(stats.t.^2 ./(stats.t.^2 + stats.dfe));
p_value = stats.p;
Network_Property = [{'Model'};NetworkPropsNames(idx)'];
se = stats.se;
unweighted_model1 = table(Network_Property,b,se,r_squared,p_value)
unweighted_model1_dfe = stats.dfe

idx = [1 2 5 6 8];
[b,~,stats] = glmfit(NetworkProperties(:,idx),Networks_mwCMC');
r_squared = sqrt(stats.t.^2 ./(stats.t.^2 + stats.dfe));
p_value = stats.p;
Network_Property = [{'Model'};NetworkPropsNames(idx)'];
se = stats.se;
unweighted_model2 = table(Network_Property,b,se,r_squared,p_value)
unweighted_model2_dfe = stats.dfe

idx = [1 2 5 7 8];
[b,~,stats] = glmfit(NetworkProperties(:,idx),Networks_mwCMC');
r_squared = sqrt(stats.t.^2 ./(stats.t.^2 + stats.dfe));
p_value = stats.p;
Network_Property = [{'Model'};NetworkPropsNames(idx)'];
se = stats.se;
unweighted_model3 = table(Network_Property,b,se,r_squared,p_value)
unweighted_model3_dfe = stats.dfe

load('Combined_Weighted_Network_results.mat')

idx = [1 5 6 7 8];
[b,~,stats] = glmfit(NetworkProperties(:,idx),Networks_mwCMC');
r_squared = sqrt(stats.t.^2 ./(stats.t.^2 + stats.dfe));
p_value = stats.p;
Network_Property = [{'Model'};NetworkPropsNames(idx)'];
se = stats.se;
weighted_model1 = table(Network_Property,b,se,r_squared,p_value)
weighted_model1_dfe = stats.dfe

idx = [1 5 6 8];
[b,~,stats] = glmfit(NetworkProperties(:,idx),Networks_mwCMC');
r_squared = sqrt(stats.t.^2 ./(stats.t.^2 + stats.dfe));
p_value = stats.p;
Network_Property = [{'Model'};NetworkPropsNames(idx)'];
se = stats.se;
weighted_model2 = table(Network_Property,b,se,r_squared,p_value)
weighted_model2_dfe = stats.dfe

idx = [1 5 7 8];
[b,~,stats] = glmfit(NetworkProperties(:,idx),Networks_mwCMC');
r_squared = sqrt(stats.t.^2 ./(stats.t.^2 + stats.dfe));
p_value = stats.p;
Network_Property = [{'Model'};NetworkPropsNames(idx)'];
se = stats.se;
weighted_model3 = table(Network_Property,b,se,r_squared,p_value)
weighted_model3_dfe = stats.dfe

save('GLMResults.mat','unweighted_model1','unweighted_model2','unweighted_model3',...
    'weighted_model1','weighted_model2','weighted_model3',...
    'unweighted_model1_dfe','unweighted_model2_dfe','unweighted_model3_dfe',...
    'weighted_model1_dfe','weighted_model2_dfe','weighted_model3_dfe');