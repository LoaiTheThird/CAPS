% ---------- Facts ----------
% 1) copper is a metal
metal(copper).

% 2) metals conduct electricity (as a category)
conducts_electricity(metal).

% 3) copper is an instance of the category metal
isa(copper, metal).

% ---------- Inference rules ----------
% Rule 1: inherit the 'conducts_electricity' property
%         from the category 'metal' to its instances.
conducts_electricity(X) :-
    isa(X, metal),
    conducts_electricity(metal).

% Rule 2: if copper conducts electricity, then our hypothesis holds.
holds(hypothesis) :-
    conducts_electricity(copper).
