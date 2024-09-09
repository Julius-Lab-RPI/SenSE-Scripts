function plot_subject_data(subject)
% Rensselaer Polytechnic Institute - Julius Lab
% SenSE Project
% Author - Chukwuemeka Osaretin Ike
%
% Description:
% Plots the subject's true JFK, CBTmins, and HR.

%% Load.
load(strcat("Data\Sim_A", num2str(subject), ".mat"), ...
        "t", "I", "x", "cbtmins", "hr", "basal_hr")
% hr = hr;
%% .
figure(1); clf;
idx = t >= 0;
subplot(2,1,1)
hold on;
plot(t(idx), x(idx,1), "LineWidth", 3)
plot(t(idx), x(idx,2), "LineWidth", 3)
hold off;
xticks(0:24:t(end))
grid on
legend("x", "x_c", "Location", "northwest")
title("JFK State Trajectories")

% subplot(3,1,2)
% plot(1:length(cbtmins), mod(12+cbtmins, 24)-12, "-o")
% ylabel("Hours after Midnight (00:00)")
% xlabel("Day")
% grid on
% xticks(0:length(cbtmins))
% title("CBT_{mins}")
% ylim([1 5])

subplot(2,1,2)
hold on
plot_light_profile(t(idx), hr(idx))
plot(t(idx), movmean(hr(idx), 60), "r-.", "LineWidth", 2)
yline(mean(hr(idx)), "k--", "LineWidth", 2, "Label", strcat(num2str(mean(hr)), " bpm"))
legend("HR", "Hourly Moving Average", "Mean HR")
title("Heart Rate")
ylabel("Heart Rate (bpm)")

fontsize(30, "points")



end