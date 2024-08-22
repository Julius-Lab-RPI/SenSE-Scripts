function [t, y] = gois09simulink(time, input, initial_conditions)
% Implementation of the Gois heart model in Simulink.
% Paper:
% An analysis of heart rhythm dynamics using a three-coupled oscillator
% model - Gois (2009).

params = [
    % Sino-Atrial parameters.
    3;      % a_sa.
    0.2;    % w_sa1.
    -1.9;   % w_sa2.
    3;      % d_sa.
    4.9;    % e_sa.

    % Atrio-Ventricular parameters.
    3;      % a_av.
    0.1;    % w_av1.
    -0.1;   % w_av2.
    3;      % d_av.
    3;      % e_av.

    % His-Purkinje complex parameters.
    5;      % a_hp.
    1;      % w_hp1.
    -1;     % w_hp2.
    3;      % d_hp.
    7;      % e_hp.

    % ECG parameters.
    1;      % alf_0.
    0.1;    % alf_1.
    0.05;   % alf_3.
    0.4;    % alf_5.

    % Coupling parameters.
    5;      % k_sa_av.
    0;      % k_sa_hp.
    0;      % k_av_sa.
    20;     % k_av_hp.
    0;      % k_hp_sa.
    0;      % k_hp_av.
];

% Variables are from the function's workspace.
options = simset('SrcWorkspace','current', 'DstWorkspace', 'current');
simulation = 'Models\Gois09.slx';

t = time;
u = input;
xInitial = [initial_conditions;
    params(16) +...
    params(17)*initial_conditions(1) +...
    params(18)*initial_conditions(3) +...
    params(19)*initial_conditions(5)
];
dt = t(2) - t(1);

out = sim(simulation, [], options); % Run simulation.
y = out.x;
t = out.tout;

end