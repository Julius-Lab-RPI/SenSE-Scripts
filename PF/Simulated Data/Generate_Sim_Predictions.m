% Rensselaer Polytechnic Institute - Julius Lab
% SenSE Project
% Author - Chukwuemeka Osaretin Ike
%
% Description:
% Generates the predicted trajectory based on the light and best practice
% initial conditions.

subject = 4;

load(strcat("Data\Sim_A", num2str(subject), ".mat"), ...
    "t", "x", "cbtmins")

% Load light.
load("Data\Sim_Light.mat", "I_pulse", "I", "I_dropout", "I_corrupted")
% I = I_dropout;

%% Simulate the model.
% Mean at midnight.
init = [-0.4068, -1.2773, 0.5102];        % x, xc, n.

% % Midnight exactly.
% init = [-0.2369, -1.3220, 0.7328];

% init = [-0.133, -0.8, 0.3];

% Generate the trajectories.
[~, u_bp, x_bp] = jewett99(t, I, init);
cbtmins_bp = getCBTMins(t, x_bp(:,1), x_bp(:,2), "jewett99");

%% Plots.
figure(1); clf;

subplot(2,2,1)
hold on
plot(t, x(:,1))
plot(t, x_bp(:,1), "--")
hold off
legend("x", "xHat", "Location", "best")
xticks(0:24:t(end))
grid on
axis padded
xlabel("Time (h)")
title(strcat("Subject ", num2str(subject), " x"))

subplot(2,2,3)
hold on
plot(t, x(:,2))
plot(t, x_bp(:,2), "--")
hold off
legend("x_c", "x_cHat", "Location", "best")
xticks(0:24:t(end))
grid on
axis padded
xlabel("Time (h)")
title("x_c")

subplot(2,2,2)
hold on;
plot(1:length(cbtmins), mod(12+cbtmins, 24)-12, "-o")
plot(1:length(cbtmins_bp), mod(12+cbtmins_bp, 24)-12, "->")
hold off
xticks([0 1:length(cbtmins)])
grid on
legend("True", "Estimate", "Location", "best")
xlabel("Day")
ylabel("Hours after Midnight (00:00)")
title("CBT_{min} Progression")


subplot(2,2,4)
hold on
plot(x(:,1), x(:,2))
plot(x_bp(:,1), x_bp(:,2), "--")
hold off
legend("True", "Estimate", "Location", "best")
xlabel("x")
ylabel("x_c")
grid on
title("Phase Portrait")

fontsize(12, "points")

%%
figure(2)
subplot(2,1,1)
plot(t, abs(x(:,1) - x_bp(:,1)))
xticks(0:24:t(end))
grid on
ylim([0, 0.6])
% axis padded
xlabel("Time (h)")
title("Absolute Error in x")

subplot(2,1,2)
plot(t, abs(x(:,2) - x_bp(:,2)))
xticks(0:24:t(end))
grid on
ylim([0, 0.6])
% axis padded
xlabel("Time (h)")
title("Absolute Error in x_c")

fontsize(12, "points")

%%
phi = wrapTo2Pi(unwrap(atan2(x(:,1), x(:,2))) - (2*pi/24)*t);
phi_bp = wrapTo2Pi(unwrap(atan2(x_bp(:,1), x_bp(:,2))) - (2*pi/24)*t);

figure(3); clf
hold on
plot(t, phi)
plot(t, phi_bp, "--")
hold off
xticks(0:24:t(end))
ylim([0 7])
grid on
% axis padded
xlabel("Time (h)")
title("\phi")




%% 
% u = u_bp;
% x = x_bp;
% cbtmins = cbtmins_bp;
% save(strcat("Data\Sim_A", num2str(subject), "_LCO_Dropout.mat"), ...
%     "t", "I", "u", "x", "cbtmins")