function generate_basal_hr(filename)
% Rensselaer Polytechnic Institute - Julius Lab
% SenSE Project
% Author - Chukwuemeka Osaretin Ike
%
% Basal HR Generator
%
% Description:
% Generates the basal heart rate oscillation based on JFK output.

%% Load JFK data.
% filename = "Data\Data_40.mat";
load(filename, "t", "x")
% x = x(1:2880, :);
% t = t(1:2880);
t = reshape(t, size(x(:,1)));

%%
init_cond = [-0.1, 0.025]';

sim_start = datetime;
[t_heart, x_heart] = sinoatrial_simulink(t, x, init_cond);
sim_end = datetime;
fprintf("Heart model runtime: %s\n", sim_end - sim_start)

%%
% figure(1); clf;
% plot(t_heart(t_heart <= 60), x_heart(t_heart <= 60,1))
% [peaks, ~] = findpeaks(x_heart(t_heart <= 60,1), "MinPeakHeight", 1);

%% Convert model output to HR.
conversion_start = datetime;
basal_hr = aggregate_peaks_per_minute(t_heart, x_heart(:,1));
fprintf("Conversion runtime: %s\n", datetime - conversion_start)

%% Save the data.
% save(filename, "basal_hr", "-append")

%% Plots.

figure(1)
subplot(2,1,1)
plot(t, x(:, 1:2))
title("JFK States"); legend("x", "x_c")
xticks(t(1):24:t(end)); grid on;

subplot(2,1,2)
plot(t, basal_hr, '-', 'LineWidth', 0.2)
hold on
plot(t, movmean(basal_hr, 10), 'r-.', 'LineWidth', 1)
yline(mean(basal_hr), 'k--', 'LineWidth', 2);
hold off
title("Heart Rate (bpm)")
xlabel("Time (h)"); ylabel("HR (bpm)")
xticks(0:24:t(end)); grid on; 
ylim([65 80])
legend("Raw Heart Rate", "10-Min Moving Average", "Mean Value")
end