function plot_pf_delta_tau(subject, run)
% Rensselaer Polytechnic Institute - Julius Lab
% SenSE Project
% Author - Chukwuemeka Osaretin Ike
%
% Description:
% Plots the PF performance for the Delta + Tau implementation.

colororder("gem12")

% subject = 10;
% run = 2;

fig = 0;

omg = 2*pi/24;

file_pref = strcat("Data\A", num2str(subject));
results_pref = strcat("Results\A", num2str(subject));

font_size = 12;

%% Load PF optimization output.
load(strcat(results_pref, "_PF_Out_Delta_Tau_", num2str(run), ".mat"))

Ns = size(weights, 1);
pf_start = 48;

%% Load subject data.
load(strcat(file_pref, "_KF.mat"), "theta", "phi")
load(strcat(file_pref, ".mat"), "t", "start_time", "I", "conv_light")
load(strcat(file_pref, "_LCO.mat"), "y_raw", "y_conv", "cbtmins_raw", "cbtmins_conv")

I = conv_light;
t = t';
t = t + start_time;
theta_kf = theta;
phi_kf = phi;
y = phi_kf;

day1 = 7;           % First day to calculate phase shift.
% Second day to calculate phase shift.
if subject == 8
    day2 = 13;
else
    day2 = 14;
end
x1 = [(day1-0.5)*24, (day1+0.5)*24, (day1+0.5)*24, (day1-0.5)*24];
x2 = [(day2-0.5)*24, (day2+0.5)*24, (day2+0.5)*24, (day2-0.5)*24];

%%
theta_raw = wrapTo2Pi(atan2(y_raw(:,1), y_raw(:,2)));
theta_conv = wrapTo2Pi(atan2(y_conv(:,1), y_conv(:,2)));
theta_pf = wrapTo2Pi(atan2(x_mean(:,1), x_mean(:,2)));

phi_raw = compute_phi(t, theta_raw, omg);
phi_conv = compute_phi(t, theta_conv, omg);
phi_pf = compute_phi(t, theta_pf, omg);

%% Plot all trajectories, mean trajectory, and true value.

% figure(1)
% subplot(4,1,1)
% 
% plot_trajectory(t, I)
% legend("Light")
% ylabel("Illuminance (lux)")
% 
% subplot(4,1,2)
% idx = 3;
% plot(t, squeeze(xHat(:,idx,:)), ":", "LineWidth", 0.01)
% hold on
% plot(t, x_mean(:,idx), "k--", "LineWidth", 2)
% hold off
% xticks(0:24:t(end))
% grid on
% title("n")
% legend([repmat("", [1, Ns]), "Mean"])
% 
% subplot(4,1,3)
% idx = 1;
% plot(t, squeeze(xHat(:,idx,:)), ":", "LineWidth", 0.01)
% hold on
% plot(t, x_mean(:,idx), "k--", "LineWidth", 2)
% hold off
% xticks(0:24:t(end))
% grid on
% title("x")
% legend([repmat("", [1, Ns]), "Mean"])
% 
% subplot(4,1,4)
% idx = 2;
% plot(t, squeeze(xHat(:,idx,:)), ":", "LineWidth", 0.01)
% hold on
% plot(t, x_mean(:,idx), "k--", "LineWidth", 2)
% hold off
% xticks(0:24:t(end))
% grid on
% title("x_c")
% legend([repmat("", [1, Ns]), "Mean"])
% 
% fontsize(16, "points")


%% Comparisons between PF and unfiltered models.

%% CBTmins.

cbtmins_pf = getCBTMins(t, movmean(x_mean(:,1), 60), x_mean(:,2), "jewett99");


figure(4); clf

idx = 1;
subplot(2,1,idx)
yl = [-1.5 1.5];
y1 = [yl(1), yl(1), yl(2), yl(2)];
fill(x1, y1, 'k', 'FaceAlpha', 0.05,  "LineStyle", "none")
hold on
fill(x2, y1, 'k', 'FaceAlpha', 0.05,  "LineStyle", "none")

plot(t, y_raw(:, idx), "-")
plot(t, y_conv(:, idx), "--")
plot(t, x_mean(:, idx), "g-.")

xline(cbtmins_raw, "b-")
xline(cbtmins_conv, "r--")
xline(cbtmins_pf, "g-.")

hold off

legend(["", "", "Raw", "Conv", "PF", ...
    "Raw", repmat("", [1, length(cbtmins_raw)-1]), ...
    "Conv", repmat("", [1, length(cbtmins_conv)-1]), ...
    "PF", repmat("", [1, length(cbtmins_pf)-1])], ...
    "Location", "northwest")
xticks(0:24:t(end)); grid on; xlabel("Time (h)")


subplot(2,1,2)
p_raw = plot(1:length(cbtmins_raw), mod(cbtmins_raw,24), "-o");
hold on
p_conv = plot(1:length(cbtmins_conv), mod(cbtmins_conv,24), "-*");
p_pf = plot(1:length(cbtmins_pf), mod(cbtmins_pf,24), "g->");
hold off

datatip(p_raw, "DataIndex", 8, "Location", "northeast");
bb = ~isnan(cbtmins_raw);
datatip(p_raw, "DataIndex", find(bb, 1, "last"), "Location", "northeast");

datatip(p_conv, "DataIndex", 8, "Location", "southwest");
bb = ~isnan(cbtmins_conv);
datatip(p_conv, "DataIndex", find(bb, 1, "last"), "Location", "southwest");

datatip(p_pf, "DataIndex", 8, "Location", "southeast");
bb = ~isnan(cbtmins_pf);
datatip(p_pf, "DataIndex", find(bb, 1, "last"), "Location", "southeast");



legend("Raw", "Conv", "PF", "Location", "northwest")
title("CBT Mins");
xlabel("Day")
ylabel("CBT_{min} (h)")
xticks(1:length(cbtmins_pf))
grid on
axis padded
% ylim([0 6])


fontsize(16, "points")

%% Plot theta and phi.

% fprintf("Mean phi raw: %3.4f\n", mean(phi_raw))
% fprintf("Mean phi conv: %3.4f\n", mean(phi_conv))
% fprintf("Mean phi pf: %3.4f\n", mean(phi_pf))
% fprintf("Mean phi kf: %3.4f\n", mean(phi_kf))

figure(5); clf;
% subplot(2,1,1)
yl = [0, 7];
y1 = [yl(1), yl(1), yl(2), yl(2)];
% fill(x1, y1, 'k', 'FaceAlpha', 0.05, "LineStyle", "none")
% hold on
% fill(x2, y1, 'k', 'FaceAlpha', 0.05, "LineStyle", "none")
% 
% plot(t, theta_kf)
% plot(t, theta_raw, "--")
% plot(t, theta_conv, "--")
% plot(t, theta_pf, "-.")
% hold off
% 
% legend("", "", "KF", "Raw", "Conv", "Filtered", "Location", "northwest")
% title("\theta")
% xticks(0:24:t(end)); grid on
% 
% 
% subplot(2,1,2)
fill(x1, y1, 'k', 'FaceAlpha', 0.05, "LineStyle", "none")
hold on
fill(x2, y1, 'k', 'FaceAlpha', 0.05, "LineStyle", "none")

plot(t, phi_kf)
plot(t, wrapTo2Pi(movmean(unwrap(phi_kf), 1440)), "r-", "LineWidth", 1)
plot(t, phi_raw, "--")
plot(t, phi_conv, "--")
plot(t, phi_pf, "g-.")
plot(t, x_mean(:,4))
hold off;

legend("", "", "KF", "Daily KF", "Raw", "Conv", "PF", "PF \Delta", "Location", "northwest")
title("\phi")
xticks(0:24:t(end)); grid on

fontsize(16, "points")

%% Plot phi, bias, and tau terms for all particles.
figure(fig+6); clf

theta_all = wrapTo2Pi(squeeze(atan2(xHat(:,1,:), xHat(:,2,:))));
phi_all = unwrap(theta_all) - omg*t;
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