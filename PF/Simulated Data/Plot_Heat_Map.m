% Rensselaer Polytechnic Institute - Julius Lab
% SenSE Project
% Author - Chukwuemeka Osaretin Ike
%
% Description:
% Plots a heat map-like graph to allow us visualize the convergence of the
% state estimates.

subject = 4;
run = 29;
num_points = 12;

font_size = 12;



file_pref = strcat("Data\Sim_A", num2str(subject));
results_pref = strcat("Results\Sim_A", num2str(subject));

% Load predicted values.
% load(strcat(file_pref, "_LCO_Dropout.mat"), "x", "cbtmins")
load(strcat(file_pref, "_LCO.mat"), "x", "cbtmins")
x_pred = x;
cbtmins_pred = cbtmins;

% Load the PF output.
load(strcat(results_pref, "_PF_Out_Delta_Tau_", num2str(run), ".mat"), "x_mean", "xHat", "weights")
x_pf = x_mean;

% Load true data.
load(strcat(file_pref, ".mat"), "t", "x", "cbtmins")

cbtmins_pf = getCBTMins(t, x_pf(:,1), x_pf(:,2), "jewett99");


%% 
figure(1); clf;
snapshots = floor(linspace(1, length(t), num_points));
for i = 1:num_points
    idx = snapshots(i);
    subplot(3,4,i)
    scatter(xHat(:, 1, idx), xHat(:, 2, idx), 500*weights(:, idx))
    % plot(xHat(:, 1, idx), xHat(:, 2, idx), "o", "MarkerSize", 4)
    hold on
    plot(x_pf(idx, 1), x_pf(idx, 2), "g^", "MarkerSize", 10)
    plot(x_pred(idx, 1), x_pred(idx, 2), "r", "Marker", "square", "MarkerSize", 10)
    plot(x(idx, 1), x(idx, 2), "ko", "MarkerSize", 10)
    hold off
    grid on
    xlabel("x"); ylabel("x_c")
    axis([-1.5 1.5 -1.5 1.5])

    if i == 4
        legend("Particles", "Filter Estimate", "Open-Loop Estimate", "True State")
    end
end
fontsize(15, "points")

