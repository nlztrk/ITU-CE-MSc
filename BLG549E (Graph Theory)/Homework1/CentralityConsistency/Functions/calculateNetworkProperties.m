function [NetProperty] = calculateNetworkProperties(A)

    Density = density_und(A);
    [~,Mgap] = majorization_gap(A);
    clustering = mean(clustering_coef_wu(A));
    assortivity = double(assortativity_wei(A,0));
    [~,efficiency] = charpath(graphallshortestpaths(sparse(A)));
    DiffEff = diffusion_efficiency(full(A));
   [~,SpecGap] = spectral_gap(double(A>0));
   
   NetProperty = [assortivity clustering Density DiffEff efficiency Mgap SpecGap];