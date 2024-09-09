function plot_pf_delta_tau(subject, run)
% Rensselaer Polytechnic Institute - Julius Lab
% SenSE Project
% Author - Chukwuemeka Osaretin Ike
%
% Description:
% Plots the PF performance for the Delta + Tau implementation.

colororder("gem12")

% subject = 5;
% run = 11;

fig = 0;

omg = 2*pi/24;

file_pref = strcat("Data\Sim_A", num2str(subject));
results_pref = strcat("Results\Sim_A", num2str(subject));

font_size = 12;

%% Load PF optimization output.
load(strcat(results_pref, "_PF_Out_Delta_Tau_", num2str(run), ".mat"))

Ns = size(weights, 1);
pf_start = 48;

%% Load subject data.
load(strcat(file_pref, "_KF.mat"), "theta", "phi")
load(strcat(file_pref, ".mat"), "t", "I", "x")
% load(strcat(file_pref, "_LCO.mat"), "x", "cbtmins")

theta_kf = theta;
phi_kf = phi;
y = phi_kf;

day1 = 7;           % First day to calculate phase shift.
day2 = 14;          % Second day to calculate phase shift.

x1 = [(day1-0.5)*24, (day1+0.5)*24, (day1+0.5)*24, (day1-0.5)*24];
x2 = [(day2-0.5)*24, (day2+0.5)*24, (day2+0.5)*24, (day2-0.5)*24];

%%
theta_lco = wrapTo2Pi(atan2(x(:,1), x(:,2)));
theta_pf = wrapTo2Pi(atan2(x_mean(:,1), x_mean(:,2)));

phi_lco = compute_phi(t, theta_lco, omg);
phi_pf = compute_phi(t, theta_pf, omg);

%% Plot all trajectories, mean trajectory, and true value.

% figure(fig+1); clf
% subplot(4,1,1)
% 
% plot(t, I, "b-", "LineWidth", 2)
% % hold on
% % plot(t, movmean(d_light, 30), "k-", "LineWidth", 2)
% xticks(0:24:t(end))
% grid on
% title("Light")
% legend("True Input")%, "PF Input")
% ylabel("Illuminance (lux)")
% 
% % xlim(xl)
% 
% subplot(4,1,2)
% idx = 3;
% plot(t, squeeze(xHat(:,idx,:)), ":", "LineWidth", 0.01)
% hold on
% plot(t, x(:,idx), "b-", "LineWidth", 2)
% plot(t, x_mean(:,idx), "k--", "LineWidth", 2)
% hold off
% xticks(0:24:t(end))
% grid on
% title("n")
% legend([repmat("", [1, Ns]), "Estimate"])
% % legend("True", "Estimate")
% 
% % xlim(xl)
% 
% subplot(4,1,3)
% idx = 1;
% plot(t, squeeze(xHat(:,idx,:)), ":", "LineWidth", 0.01)
% hold on
% plot(t, x(:,idx), "b-", "LineWidth", 2)
% plot(t, x_mean(:,idx), "k--", "LineWidth", 2)
% hold off
% xticks(0:24:t(end))
% grid on
% title("x")
% legend([repmat("", [1, Ns]), "True", "Estimate"])
% % legend("True", "Estimate")
% 
% % xlim(xl)
% 
% subplot(4,1,4)
% idx = 2;
% plot(t, squeeze(xHat(:,idx,:)), ":", "LineWidth", 0.01)
% hold on
% plot(t, x(:,idx), "b-", "LineWidth", 2)
% plot(t, x_mean(:,idx), "k--", "LineWidth", 2)
% hold off
% xticks(0:24:t(end))
% grid on
% title("x_c")
% legend([repmat("", [1, Ns]), "True", "Estimate"])
% % legend("True", "Estimate")
% 
% % xlim(xl)
% xlabel("Time (h)")
% 
% fontsize(font_size, "points")


%% Plot theta and phi.

% fprintf("Mean phi raw: %3.4f\n", mean(phi_lco))
% fprintf("Mean phi pf: %3.4f\n", mean(phi_pf))
% fprintf("Mean phi kf: %3.4f\n", mean(phi_kf))
% 
% figure(fig+5); clf;
% % subplot(2,1,1)
% yl = [0, 7];
% y1 = [yl(1), yl(1), yl(2), yl(2)];
% % fill(x1, y1, 'k', 'FaceAlpha', 0.05, "LineStyle", "none")
% % hold on
% % fill(x2, y1, 'k', 'FaceAlpha', 0.05, "LineStyle", "none")
% % 
% % plot(t, theta_kf)
% % plot(t, theta_raw, "--")
% % plot(t, theta_conv, "--")
% % plot(t, theta_pf, "-.")
% % hold off
% % 
% % legend("", "", "KF", "Raw", "Filtered", "Location", "best")
% % title("\theta")
% % xticks(0:24:t(end)); grid on
% % 
% % 
% % subplot(2,1,2)
% fill(x1, y1, 'k', 'FaceAlpha', 0.05, "LineStyle", "none")
% hold on
% fill(x2, y1, 'k', 'FaceAlpha', 0.05, "LineStyle", "none")
% 
% plot(t, phi_kf)
% plot(t, wrapTo2Pi(movmean(unwrap(phi_kf), 1440)), "r-", "LineWidth", 1)
% plot(t, phi_lco, "m--", "LineWidth", 2)
% plot(t, phi_pf, "g-.")
% plot(t, x_mean(:,4))
% hold off;
% 
% legend("", "", "KF", "Daily KF", "True", "PF", "PF \Delta", "Location", "best")
% title("\phi")
% xticks(0:24:t(end)); grid on
% 
% fontsize(font_size, "points")

%% Plot phi, bias, and tau terms for all particles.
figure(fig+6); clf

theta_all = wrapTo2Pi(squeeze(atan2(xHat(:,1,:), xHat(:,2,:))));
phi_all = unwrap(theta_all) - omg*t';
phi_all = wrapTo2Pi(phi_all);

% subplot(3,1,1)
% plot(t(1), 0)
% hold on
% plot(t, theta_all(:,:), ":", "LineWidth", 0.01)
% plot(t, theta_pf, "k--")
% hold off
% xticks(0:24:t(end))
% grid on
% title("\theta")
% legend(["", repmat("", [1, Ns]), "Mean"])

subplot(3,1,1)
plot(t(1), 0)
hold on
plot(t, phi_all(:,:), ":", "LineWidth", 0.01)
plot(t, phi_pf, "k-", "LineWidth", 2)
hold off
ylim([0 7])
xticks(0:24:t(end))
grid on
title("\phi")
legend(["", repmat("", [1, Ns]), "Mean"])


subplot(3,1,2)
idx = 4;
plot(t(1), 0)
hold on
plot(t, wrapToPi(squeeze(xHat(:,idx,:))), ":", "LineWidth", 0.01)
plot(t, wrapToPi(x_mean(:,idx)), "k-", "LineWidth", 2)
hold off
ylim([-3.5 3.5])
xticks(0:24:t(end))
grid on
title("\Delta")
legend(["", repmat("", [1, Ns]), "Mean"])

subplot(3,1,3)
idx = 5;
hold on
plot(t, squeeze(xHat(:,idx,:)), ":", "LineWidth", 0.01)
plot(t, x_mean(:,idx), "k-", "LineWidth", 2)
hold off
% ylim([-3.5 3.5])
xticks(0:24:t(end))
grid on
title("\tau")
legend([repmat("", [1, Ns]), "x_{mean}"])


fontsize(font_size, "points")