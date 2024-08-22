function [tout, x] = vdp_simulink(t, input, initial_conditions)
% Simple Simulink implementation of a Van der Pol oscillator.

params = [
    0.75;   % eps.
    2;      % a.
    2*pi/24*1.3;% w.
    0.3;    % d.
];

% Variables are from the function's workspace.
options = simset('SrcWorkspace','current', 'DstWorkspace', 'current');
simulation = 'Models\VDP.slx';

out = sim(simulation, [], options); % Run simulation.
tout = out.tout;
x = out.x;

end