% Rensselaer Polytechnic Institute - Julius Lab
% SenSE Project
% Author - Chukwuemeka Osaretin Ike
%
% Description:
% Generates a light profile and saves it to a file.

%% Generate light.
num_days = 15;
shift_day = 8;
% shift = 0;
% t_std = 0;
% I_max_std = 0;
% I_low_std = 0;
% switch_std = 0;
% corrupt_light = false;
shift = 1.5;
t_std = 1.5;
I_max_std = 300;
I_low_std = 10;
switch_std = 1;
corrupt_light = true;

[~, I_pulse] = generate_light_profile(num_days, 0, 0, 0, 0, 0, 0, false);

[t, I] = generate_light_profile(num_days, shift_day, shift, ...
    t_std, I_max_std, I_low_std, switch_std, corrupt_light);
t = t';


% Create dropped out light.
I_dropout = dropout_signal(I, 120);
I_corrupted = corrupt_daily_light(I, 0, 0);

%% Plot light.
figure(1); clf;

subplot(4,1,1)
plot_light_profile(t, I_pulse)
title("A")
xline(shift_day*24, "--", "LineWidth", 2)
legend("Light", "Shift Day")
ylim([0 1800])

subplot(4,1,2)
plot_light_profile(t, I)
title("B")
xline(shift_day*24, "--", "LineWidth", 2)
legend("Light", "Shift Day")
ylim([0 1800])

subplot(4,1,3)
plot_light_profile(t, I_corrupted)
title("C")
xline(shift_day*24, "--", "LineWidth", 2)
legend("Light", "Shift Day")
ylim([0 1800])

subplot(4,1,4)
plot_light_profile(t, I_dropout)
title("D")
xline(shift_day*24, "--", "LineWidth", 2)
legend("Light", "Shift Day")
ylim([0 1800])

xlabel("Time (h)")

%% Save light.
% save("Data\Sim_Light.mat", "t", "I_pulse", "I", "I_dropout", "I_corrupted", ...
%     "num_days", "shift_day", "shift", "t_std", ...
%     "I_max_std", "I_low_std", "switch_std", "corrupt_light")