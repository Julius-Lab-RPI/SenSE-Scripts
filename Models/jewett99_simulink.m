function [t, u, y] = jewett99_simulink(time, input, initial_conditions)
% Paper:
% Revised limit cycle oscillator model of human circadian pacemaker -
% Jewett (1999).

% Process L Parameters.
a0 = 0.16;
p = 0.6;
beta = 0.013;
G = 19.875;

% I have no clue why, but Serkh (2014) and Jiawei Yin (2020) both use these
% parameters. Jiawei cites Serkh, who cites Jewett (1999), but her paper
% uses the parameters above. I have these here for clarity I guess.
% Running simulations with both obviously yields different qualitative
% behavior.
% a0 = 0.05;
% p = 0.5;
% beta = 0.0075;
% G = 33.75;

I0 = 9500;
processLParams = [a0, p, beta, G, I0];

% Process P Parameters.
mu = 0.13;
k = 0.55;
q = 1/3;
kc = 0.4;
tau = 24.2;
processPParams = [mu, k, q, kc, tau];


% Variables are from the function's workspace.
options = simset('SrcWorkspace','current', 'DstWorkspace', 'current');
simulation = 'Models\Jewett99.slx';

I = input;
t = time;
dt = time(2) - time(1);
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