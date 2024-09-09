% Rensselaer Polytechnic Institute - Julius Lab
% SenSE Project
% Author - Chukwuemeka Osaretin Ike
%
% Description:
% Generates an activity profile and saves it to a file.

%% Generate activity.
num_days = 15;
shift_day = 8;
shift = 1.5;
t_std = 1.5;
a_high_std = 10;
a_low_std = 5;
switch_std = 1;
corrupt_activity = true;

[~, a_pulse] = generate_activity_profile(num_days, shift_day, shift, 0, 0, 0, 0, false);
[t, a] = generate_activity_profile(num_days, shift_day, shift, ...
    t_std, a_high_std, a_low_std, switch_std, corrupt_activity);

% Create corrupted and dropped out activity.
a_dropout = dropout_signal(a, 120);
a_corrupted = corrupt_activity_signal(a);

%% Plots.
figure(1); clf;

subplot(4,1,1)
plot_light_profile(t, a_pulse)
title("A")
xline(shift_day*24, "--", "LineWidth", 2)
legend("Light", "Shift Day")
ylim([0 100])

subplot(4,1,2)
plot_light_profile(t, a)
title("B")
xline(shift_day*24, "--", "LineWidth", 2)
legend("Light", "Shift Day")
ylim([0 100])

subplot(4,1,3)
plot_light_profile(t, a_corrupted)
title("C")
xline(shift_day*24, "--", "LineWidth", 2)
legend("Light", "Shift Day")
ylim([0 100])

subplot(4,1,4)
plot_light_profile(t, a_dropout)
title("D")
xline(shift_day*24, "--", "LineWidth", 2)
legend("Light", "Shift Day")
ylim([0 100])

xlabel("Time (h)")

%% Save.
% save("Data\Sim_Activity.mat", "t", "a_pulse", "a", "a_dropout", "a_corrupted", ...
%     "num_days", "shift_day", "shift", "t_std", ...
%     "a_high_std", "a_low_std", "switch_std", "corrupt_activity")