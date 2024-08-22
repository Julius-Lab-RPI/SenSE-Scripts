function [t, y] = sinoatrial_simulink(time, jfk_x, initial_conditions)

params = [
    % Sino-Atrial parameters.
    40;      % a1.
    0.83;    % u11.
    -0.83;   % u12.
    3;      % d1.
    3.5;    % e1.
    22;     % f1.
    3.1;   % k_jfk_sa.
];

t = time*3600;
xInitial = initial_conditions;
dt = t(2) - t(1);

% Variables are from the function's workspace.
options = simset('SrcWorkspace','current', 'DstWorkspace', 'current');
simulation = 'Models\Sinoatrial.slx';

out = sim(simulation, [], options); % Run simulation.
t = out.t;
y = out.x;

end