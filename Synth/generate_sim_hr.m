function generate_sim_hr(subject)
% Rensselaer Polytechnic Institute - Julius Lab
% SenSE Project
% Author - Chukwuemeka Osaretin Ike
%
% Description:
% Combines the basal HR, activity, and cortisol to generate a measured HR
% for a specified subject

% subject = 1;

%% Load.
filename = strcat("Data\Sim_A", num2str(subject), ".mat");
load(filename, "basal_hr", "t")
load("Data\Sim_Activity.mat", "a")
load("Data\Sim_Cortisol.mat", "eps_hr")

%% Combine.
d = 0.3;    % Increase in HR per unit activity.

hr = basal_hr + (d*a) + eps_hr;

%% Plot.
figure(4); clf;

subplot(4,1,1)
plot_light_profile(t, basal_hr)
ylabel("Heart Rate (bpm)")
title("Basal HR")

subplot(4,1,2)
plot_light_profile(t, a)
ylabel("Activity (steps/min)")
title("Step Count")

subplot(4,1,3)
plot_light_profile(t, eps_hr)
ylabel("Heart Rate (bpm)")
title("HR Noise")

subplot(4,1,4)
plot_light_profile(t, hr)
hold on
plot(t, movmean(hr, 60), "r--", "LineWidth", 2)
hold off
ylabel("Heart Rate")
title("Combined HR")
ylim([20 120])

%% Save.
save(filename, "hr", "-append")
end