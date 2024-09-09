% Rensselaer Polytechnic Institute - Julius Lab
% SenSE Project
% Author - Chukwuemeka Osaretin Ike
%
% Description:
% Generates a trajectory and CBTmins based on loaded light data.

subject = 4;

%% Light trajectory.

% % Generate light.
% num_days = 15;
% shift_day = 8;
% % shift = 0;
% % t_std = 0;
% % I_max_std = 0;
% % I_low_std = 0;
% % switch_std = 0;
% % corrupt_light = false;
% shift = 1.5;
% t_std = 1.5;
% I_max_std = 300;
% I_low_std = 10;
% switch_std = 1;
% corrupt_light = true;
% 
% [t, I] = generate_light_profile(num_days, shift_day, shift, ...
%     t_std, I_max_std, I_low_std, switch_std, corrupt_light);

% Load light.
load("Data\Sim_Light.mat", "t", "I_pulse", "I", "I_dropout", "I_corrupted")
% I = I_pulse;

%% Simulate the model.
% init = [-0.6067, -1.053, 0.3718];       % Subject 1.
% init = [-0.7312, -0.9815, 0.3717];      % Subject 2.
% init = [-0.5274, -1.0875, 0.3717];      % Subject 3.
% init = [-0.8503, -0.8853, 0.3716];      % Subject 4.
% init = [-0.4536, -1.1119, 0.3719];      % Subject 5.
init = [-0.7782, -0.9419, 0.3718];      % Subject 6.

% Generate the trajectories.
[~, u, x] = jewett99(t, I, init);
cbtmins = getCBTMins(t, x(:,1), x(:,2), "jewett99");

%% Plots.
figure(1); clf;

subplot(2,2,1)
plot(t, I)
xticks(0:24:t(end))
grid on
axis padded
xlabel("Time (h)")
title(strcat("Subject ", num2str(subject), " Light"))
ylabel("Illuminance (lux)")

subplot(2,2,3)
plot(t, x(:,1), t, x(:,2))
xline(cbtmins, "--")
legend("x", "xc", "CBT_{min}")
xticks(0:24:t(end))
grid on
axis padded
xlabel("Time (h)")
title("State Trajectories")

subplot(2,2,2)
plot(1:length(cbtmins), mod(12+cbtmins, 24)-12, "->")
xticks([0 1:length(cbtmins)])
grid on
xlabel("Day")
ylabel("Hours after Midnight (00:00)")
title("CBT_{min} Progression")

subplot(2,2,4)
plot(x(:,1), x(:,2))
xlabel("x")
ylabel("x_c")
grid on
title("Phase Portrait")

fontsize(12, "points")


%% Save the trajectories and parameters.
% save(strcat("Data\Sim_A", num2str(subject), ".mat"), ...
%     "t", "I", "u", "x", "cbtmins", "init", ...
%     "num_days", "shift_day", "shift", "t_std", "I_max_std", "I_low_std", ...
%     "switch_std", "corrupt_light")