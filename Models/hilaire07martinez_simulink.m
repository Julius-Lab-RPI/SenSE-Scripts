function [t, u, y] = hilaire07martinez_simulink(time, input, params, initial_conditions)
% Implementation of St Hilaire's model (parameters and light-sensitivity),
% but without the non-photic component. This version was used in
% Vicente-Martinez (2024).

% Paper:
% Addition of a non-photic component to a light-based mathematical model of
% the human circadian pacemaker - St Hilaire (2007).

% Modifided in:
% Uncovering personal circadian responses to light through particle swarm
% optimization - Vicente-Martinez (2024).

% This model also has tunable params.
% params = [tau; p; k].
% initial_conditions = [x0; xc0; n0].

% Process L Parameters.
a0 = 0.1;
% p = 0.5;
p = params(2);
beta = 0.007;
G = 37;
I0 = 9500;
processLParams = [a0, p, beta, G, I0];

% Process P Parameters.
mu = 0.13;
% k = 0.55;
k = params(3);
q = 1/3;
kc = 0.4;
% tau = 24.2;
tau = params(1);
processPParams = [mu, k, q, kc, tau];

% Variables are from the function's workspace.
options = simset('SrcWorkspace','current', 'DstWorkspace', 'current');
simulation = 'Models\Hilaire07Martinez.slx';

I = input;
t = time;
dt = t(2) - t(1);
xInitial = initial_conditions(1:2);
nInitial = initial_conditions(3);

out = sim(simulation, [], options); % Run simulation.
x = out.xOut(:, 1);
xc = out.xOut(:, 2);
n = out.nOut;
t = out.tout;
u = out.uOut;
y = [x, xc, n];
end