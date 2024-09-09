function generate_model_trajectory(subject)
% Rensselaer Polytechnic Institute - Julius Lab
% SenSE Project
% Author - Chukwuemeka Osaretin Ike
%
% Description:
% Loads the data for a given subject and runs the model with their initial
% conditions to create the trajectory. Saves the trajectory and CBT mins to
% the same file.

%% .
% Load the light inputs.
vars = {"t", "I", "conv_light", "start_time", "init"};
load(strcat("Data\A", num2str(subject), ".mat"), vars{:})
t = t+start_time;

% Run the model on raw light.
% [t_raw, u_raw, y_raw] = jewett99_rk4(t, I, init);
[t_raw, u_raw, y_raw] = jewett99(t, I, init);
cbtmins_raw = getCBTMins(t_raw, y_raw(:,1), y_raw(:,2), "jewett99");

% Run the model on converted light.
% [t_conv, u_conv, y_conv] = jewett99_rk4(t, conv_light, init);
[t_conv, u_conv, y_conv] = jewett99(t, conv_light, init);
cbtmins_conv = getCBTMins(t_conv, y_conv(:,1), y_conv(:,2), "jewett99");

%% Plots.

figure(1); clf
subplot(3,1,1)
plot(t_raw, y_raw(:,1), t_raw, y_raw(:,2))
xline((cbtmins_raw))
xticks(0:24:t_raw(end))
xlabel("Time (h)")
title("State Trajectories")

subplot(3,1,2)
plot(t_conv, y_conv(:,1), t_conv, y_conv(:,2))
xline((cbtmins_conv))
xticks(0:24:t_conv(end))
xlabel("Time (h)")
title("State Trajectories")

subplot(3,1,3)
scatter(1:length(cbtmins_raw), mod(cbtmins_raw, 24))
hold on;
scatter(1:length(cbtmins_conv), mod(cbtmins_conv,24))
hold off;
legend("Raw Light", "Converted Light")
xlabel("Day")
title("CBT_{min} Progression")

%% Save the trajectories and CBT mins.
filename = strcat("Data\A", num2str(subject), "_LCO_Hilaire.mat");
save(filename, ...
    "t_raw", "u_raw", "y_raw", "cbtmins_raw", ...
    "t_conv", "u_conv", "y_conv", "cbtmins_conv")
end