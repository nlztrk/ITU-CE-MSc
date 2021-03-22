# CentralityConsistency

The files in this directory are able to rerun all analyses performed in Oldham et al., 2019. Consistency and differences between centrality measures across distinct classes of networks.

Paper can be found [here](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0220061)

If you have questions please contact Stuart by [email](mailto:stuart.oldham@monash.edu)

Dependencies:
[Brain Connectivity Toolbox](https://sites.google.com/site/bctnet/), Version 2017-15-01

[MatlabBGL](https://au.mathworks.com/matlabcentral/fileexchange/10922-matlabbgl), Version 1.1.0.0

MainAnalysisScript.m will run all analyses performed in the paper. Running this script as is will take a very long time especially for the running of centrality measures and generation of nulls/surrogate networks. If you have access to a cluster I recommend breaking up the scirpt and splitting the analysis of individual networks across computers/nodes.

FigureGeneration.m will reproduce all figures from the paper.

The data used in the paper can be found [here](https://figshare.com/s/22c5b72b574351d03edf). Simply make sure it is added to the path (can just simply extract the data into the directory) so data can be accessed.
