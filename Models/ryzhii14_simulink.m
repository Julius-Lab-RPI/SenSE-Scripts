function [t, y] = ryzhii14simulink(time, input, initial_conditions)
% 
% Papers:
% An analysis of heart rhythm dynamics using a three-coupled oscillator
% model - Gois (2009).
% A heterogeneous coupled oscillator model for simulation of
% ECG signals - Ryzhii (2014).

params = [
    % SA Node.
    40; % p.a1
    0.83; % p.u11
    -0.83; % p.u12
    22; % p.f1
    3; % p.d1
    3.5; % p.e1

    % AV Node.
    50; % p.a2
    0.83; % p.u21
    -0.83; % p.u22
    8.4; % p.f2
    3; % p.d2
    5; % p.e2

    % HP Complex.
    50; % p.a3
    0.83; % p.u31
    -0.83; % p.u32
    1.5; % p.f3
    3; % p.d3
    12; % p.e3

    % Coupling Params.
    22; % p.k_sa_av
    22; % p.k_av_hp

    % ECG parameters.
    1; % p.alf_0.
    0.06; % p.alf_1.
    0.1; % p.alf_3.
    0.3; % p.alf_5.
];

% Variables are from the function's workspace.
options = simset('SrcWorkspace','current', 'DstWorkspace', 'current');
simulation = 'Models\Ryzhii14.slx';

t = time;
u = input;
xInitial = [initial_conditions;
    params(21) +...
    params(22)*initial_conditions(1) +...
    params(23)*initial_conditions(3) +...
    params(24)*initial_conditions(5)
];
dt = t(2) - t(1);

out = sim(simulation, [], options); % Run simulation.
y = out.x;

end