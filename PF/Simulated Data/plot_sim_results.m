function plot_sim_results(subject, run, method, comparison)
% Rensselaer Polytechnic Institute - Julius Lab
% SenSE Project
% Author - Chukwuemeka Osaretin Ike
%
% Description:
% Plots the trajectories for the true, predicted, and PF data. Start of the
% comparison.

% subject = 4;
% run = 11;

fig = 10;
font_size = 12;
omg = (2*pi/24);

file_pref = strcat("Data\Sim_A", num2str(subject));
results_pref = strcat("Results\Sim_A", num2str(subject));

%% Load data.
% Load predicted values.
if comparison == "dropout"
    load(strcat(file_pref, "_LCO_Dropout.mat"), "x", "cbtmins")
else
    load(strcat(file_pref, "_LCO.mat"), "x", "cbtmins")
end
x_pred = x;
cbtmins_pred = cbtmins;

% Load true data.
load(strcat(file_pref, ".mat"), "t", "I", "x", "cbtmins")

% Load KF phi.
load(strcat(file_pref, "_KF.mat"), "phi")
phi_kf = wrapTo2Pi(phi)/omg;

% Load the PF output.
if method == "tau"
    load(strcat(results_pref, "_PF_Out_Delta_Tau_", num2str(run), ".mat"), "x_mean", "xHat")
else
    load(strcat(results_pref, "_PF_Out_Delta_", num2str(run), ".mat"), "x_mean", "xHat")
end
x_pf = x_mean;
clear x_mean
cbtmins_pf = getCBTMins(t, x_pf(:,1), x_pf(:,2), "jewett99");

%% Plots.
figure(fig + 1); clf;

% subplot(3,1,1)
% hold on
% plot(t, x(:,1))
% plot(t, x_pred(:,1), "r-.")
% plot(t, x_pf(:,1), "g--")
% hold off
% legend("x", "x_{pred}", "x_{pf}", "Location", "best")
% xticks(0:24:t(end))
% grid on
% axis padded
% xlabel("Time (h)")
% title(strcat("Subject ", num2str(subject), " x"))
% 
% subplot(3,1,2)
% hold on
% plot(t, x(:,2))
% plot(t, x_pred(:,2), "r-.")
% plot(t, x_pf(:,2), "g--")
% hold off
% legend("xc", "xc_{pred}", "xc_{pf}", "Location", "best")
% xticks(0:24:t(end))
% grid on
% axis padded
% xlabel("Time (h)")
% title("xc")

% subplot(3,1,3)
subplot(2,1,1)
hold on;
plot(1:length(cbtmins), mod(12+cbtmins, 24)-12, "-o")
plot(1:length(cbtmins_pred), mod(12+cbtmins_pred, 24)-12, "r-*")
plot(1:length(cbtmins_pf), mod(12+cbtmins_pf, 24)-12, "g->")
hold off
xticks([0 1:length(cbtmins)+1])
ylim([0 6])
grid on
legend("True", "Predicted", "PF", "Location", "best")
xlabel("Day")
ylabel("Hours after Midnight (00:00)")
title("CBT_{min} Progression")

subplot(2,1,2)
hold on;
plot(1:length(cbtmins), abs(mod(12+cbtmins, 24)-12 - (mod(12+cbtmins_pred, 24)-12)), "r-*")
plot(1:length(cbtmins_pf), abs(mod(12+cbtmins, 24)-12 - (mod(12+cbtmins_pf, 24)-12)), "g->")
hold off
xticks([0 1:length(cbtmins)+1])
ylim([0 3])
grid on
legend("Predicted", "PF", "Location", "best")
xlabel("Day")
ylabel("Hours after Midnight (00:00)")
title("Daily Error in CBT_{min}")

fontsize(font_size, "points")

%%
figure(fig + 2); clf;

subplot(2,1,1)
hold on;
plot(t, abs(x(:,1) - x_pred(:,1)), "r-.")
plot(t, abs(x(:,1) - x_pf(:,1)), "g--")
hold off
legend("Pred", "PF")
xticks(0:24:t(end))
grid on
ylim([0, 1])
% axis padded
xlabel("Time (h)")
title("Absolute Error in x")

subplot(2,1,2)
hold on;
plot(t, abs(x(:,2) - x_pred(:,2)), "r-.")
plot(t, abs(x(:,2) - x_pf(:,2)), "g--")
hold off
legend("Pred", "PF")
xticks(0:24:t(end))
grid on
ylim([0, 1])
% axis padded
xlabel("Time (h)")
title("Absolute Error in x_c")

fontsize(font_size, "points")

%%
phi = (wrapTo2Pi(unwrap(atan2(x(:,1), x(:,2))) - omg*t))/omg;
phi_pred = (wrapTo2Pi(unwrap(atan2(x_pred(:,1), x_pred(:,2))) - (omg*t)))/omg;
phi_pf = (wrapTo2Pi(unwrap(atan2(x_pf(:,1), x_pf(:,2))) - omg*t))/omg;

% figure(fig+3); clf
% hold on
% plot(t, phi)
% plot(t, phi_pred)
% plot(t, phi_pf)
% plot(t, x_pf(:,4)/omg)
% plot(t, phi_kf)
% plot(t, movmean(phi_kf, 1440))
% hold off
% legend("True", "Pred", "PF", "PF \Delta", "KF")
% yticks(0:2:24)
% xticks(0:24:t(end))
% grid on
% ylim([0 24])
% % axis padded
% xlabel("Time (h)")
% title("\phi")


%% Plot particle evolutions.

theta_all = wrapTo2Pi(squeeze(atan2(xHat(:,1,:), xHat(:,2,:))));
phi_all = unwrap(theta_all) - omg*t';
phi_all = wrapTo2Pi(phi_all)/omg;
Ns = size(xHat, 1);

figure(fig+4); clf

subplot(3,1,1)
plot(t(1), 0)
hold on
plot(t, phi_all(:,:), ":", "LineWidth", 0.01)
plot(t, phi_pf, "k-", "LineWidth", 2)
plot(t, phi, "b", "LineWidth", 2)
hold off
ylim([0 24])
yticks(0:3:24)
xticks(0:24:t(end))
ylabel("Phase Offset (h)")
grid on
title("Phase Offset \phi")
legend(["", repmat("", [1, Ns]), "Filter Estimate", "True Offset"])


subplot(3,1,2)
idx = 4;
plot(t(1), 0)
hold on
plot(t, wrapToPi(squeeze(xHat(:,idx,:)))/omg, ":", "LineWidth", 0.01)
plot(t, wrapToPi(x_pf(:,idx))/omg, "k-", "LineWidth", 2)
hold off
ylim([-12 12])
yticks(-12:3:12)
ylabel("Phase Bias (h)")
xticks(0:24:t(end))
grid on
title("\Delta")
legend(["", repmat("", [1, Ns]), "Filter Estimate"])

if length(x_pf(1,:)) == 5
    subplot(3,1,3)
    hold on;
    plot(t, squeeze(xHat(:, 5, :)), ":", "LineWidth", 0.01)
    plot(t, x_pf(:, 5), "k", "LineWidth", 2)
    % yline(23.95, "b--", "LineWidth", 2)
    hold off
    xticks(0:24:t(end))
    grid on
    xlabel("Time (h)")
    ylabel("Intrinsic Period (h)")
    title("Intrinsic Period \tau")
    legend([repmat("", [1, Ns]), "Filter Estimate"], "Location","west")%, "True Period"])
end

fontsize(font_size, "points")