% makes figure 5 and 6

varsbefore = who;

load('Combined_Unweighted_surrogate_results.mat')

Cmap = [121 85 72; 3 168 243; 74 174 78; 103 58 182; 176 176 176; 252 133 95]/255;
for j = 1:2

if j == 1
    surrogate_type = 'unconstrained';
    printname = 'Figure5.tif';
else
    surrogate_type = 'constrained';
    printname = 'Figure6.tif';
end
for i = 1:length(Networks)
    % Calculate difference between surrogates and empirical network
    % properties and CMCs
    CMCnorm(i) = (Networks_mwCMC(i) - mean(NullNetworks_mwCMC{i,j}));
    NullMean(i,:) = mean(NullNetProperty{i,j});
    Nullstd(i,:) = std(NullNetProperty{i,j});
   
end
figure('units','pixels','outerposition',[0 0 2560 1440])
for k = 1:8
    subplot(2,4,k)
    if k == 3
        scatter(NetworkProperties(:,k),CMCnorm,50,Type,'filled')
        xlabel(NetworkPropsNames{k})
    elseif j == 2 && k == 6
        scatter(NetworkProperties(:,k),CMCnorm,50,Type,'filled')
        xlabel(NetworkPropsNames{k})
    else
    Propnorm = (NetworkProperties(:,k) - NullMean(:,k));
    
    scatter(Propnorm,CMCnorm,50,Type,'filled')
    colormap(Cmap)
    xlabel({['Empirical ',NetworkPropsNames{k},'-'],[surrogate_type,' ',NetworkPropsNames{k}]})
    colormap(Cmap)
    caxis([1 6])
    if k == 1 || k == 5
    ylabel({'Empirical mean CMC -',[surrogate_type,' mean CMC']});
    end
    end
    set(gca,'Fontsize',16)
end
print(printname,'-dtiff','-r300')
end

varsafter = who; 
varsnew = setdiff(varsafter, varsbefore); 
clear(varsnew{:})