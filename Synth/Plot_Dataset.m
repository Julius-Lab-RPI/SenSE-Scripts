% Rensselaer Polytechnic Institute - Julius Lab
% SenSE Project
% Author - Chukwuemeka Osaretin Ike
%
% Description:
% Plots the entire dataset - JFK's, CBTmins, and HR.

subjects = 1:5;
num_subjects = length(subjects);
omg = 2*pi/24;

titles = ["JFK Trajectory", "CBT_{mins}", "Heart Rate", "Phase Offset \phi"];
ylabels = ["", "Hours after Midnight (00:00)", "Heart Rate (bpm)", "Phase Offset (h)"];
% taus = ["24.2", "24.05", "24.3", "23.9167", "24.4"];
taus = ["24h12m", "24h03m", "24h18m", "23h55m", "24h24m"];

markers = ["-o", "->", "-*", "-^", "-+"];
colors = ["b", "r", "g", "k", "m"];

figure(2); clf; hold on;
figure(4); clf; hold on;

for i = 1:num_subjects
    subject = subjects(i);
    % Load subject's data.
    load(strcat("Data\Sim_A", num2str(subject), ".mat"), ...
            "t", "x", "cbtmins", "hr")
    theta_lco = atan2(x(:,1), x(:,2));
    phi_lco = compute_phi(t, theta_lco, omg);
    % phi_lco = movmean(phi_lco, 1440);
    load(strcat("Data\Sim_A", num2str(subject), "_KF.mat"), "phi")
    % phi_kf = movmean(phi, 1440);
    phi_kf = phi;

    % % JFK trajectory.
    % figure(1);
    % subplot(num_subjects, 1, i)
    % plot_light_profile(t, x(:, 1))
    % hold on
    % plot_light_profile(t, x(:, 2))
    % hold off
    % ylabel(ylabels(1))
    % legend("x", "x_c", "Location", "northwest")
    % 
    % if i == 1
    %     title(titles(1))
    % end

    % CBTmins.
    figure(2)
    plot(1:length(cbtmins), mod(12+cbtmins, 24)-12, ...
        strcat(colors(i), markers(i)), ...
         "LineWidth", 2, "MarkerSize", 20, ...
        "DisplayName", strcat("Subject ",num2str(subject), ", \tau = ", taus(i)))

    % Heart Rate.
    figure(3);
    subplot(num_subjects, 1, i)
    plot_light_profile(t, hr)
    hold on
    plot_light_profile(t, movmean(hr, 60))
    [~, mins] = findpeaks(-movmean(hr, 60),"MinPeakDistance", 720);
    xline(t(mins))
    hold off
    legend("HR", "1-Hour Movmean")
    ylabel(ylabels(3))
    if i == 1
        title(titles(3))
    end

    % Plot the KF and JFK Phi, and the Delta between the two.
    figure(4);
    plot(t, phi_lco/omg, colors(i), "DisplayName", strcat("Subject ", num2str(subject), " True \phi"))
    plot(t, phi_kf/omg, colors(i), "DisplayName", strcat("Subject ", num2str(subject), " KF \phi"))
    plot(t, wrapTo2Pi(phi_lco-phi_kf)/omg, colors(i), "DisplayName", strcat("Subject ", num2str(subject), " \Delta"))

end
%%
figure(2);
hold off;
ylabel(ylabels(2))
xlabel("Day")
grid on
xticks(0:16)
title(titles(2))
ylim([1 5])
xlim([0 16])
legend
fontsize(30, "points")


figure(4);
hold off;
ylabel(ylabels(4))
xlabel("Day")
grid on
yticks(0:2:24)
xticks(0:24:t(end))
title(titles(4))
legend
fontsize(25, "points")