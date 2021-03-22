function [SGAP, SGAP_norm] = spectral_gap(A)

V = eig(full(A));

E = sort(V,'descend');

SGAP = E(1) - E(2);

SGAP_norm = 1-(E(2)/E(1));