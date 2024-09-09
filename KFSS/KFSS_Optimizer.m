% Rensselaer Polytechnic Institute - Julius Lab
% SenSE Project
% Author - Chukwuemeka Osaretin Ike
%
% Steady-State Kalman Filter Optimization.
%
% Description:
% This script optimizes the SSKF Estimator once on the input data.

% Start the overall clock to see how long optimization takes.
startAll = clock;

% Optimization Hyperparameters
iterations = 100;            % Maximum iterations.
mu = 100;                   % Number of every generation.
lambda = 50;                % Number of children.
rho = 2;                    % Number of parents to create one child.

% Optimization bounds.
LB = -5;                % Q Lower bound - 10^LB.
lEnd = 0;               % Q lower end.
rEnd = 18;              % Twice the number of orders of magnitudes.
rLA = 1e2;              % R Lower bound.
rLB = 1e8;              % R upper bound. 

% **********************************************************************
% Load the data.

% % Synthetic data.
% load('Data/jfkDataShift8.mat', 'time', 'output')
% t = time;   y = output(:,1);
% day1 = 9;           % First day to calculate phase shift
% day2 = 25;          % Second day to calculate phase shift

% Real data.
subject = 6;
fprintf("Subject %d\n", subject)
[t, y, I, steps] = Utils.loadRealData(subject);
% y = steps;
day1 = 7;           % First day to calculate phase shift.
% Second day to calculate phase shift.
if subject == 8
    day2 = 13;
else
    day2 = 14;
end
% **********************************************************************

% Take the FFT of original signal for cost calculation
[originalSpectrum, f, len] = Utils.computeSpectrum(t, y);
originalSpectrum = originalSpectrum';

order = 1;
fprintf("Filter Order %d\n", order)
stateLength = (2*order + 1);        % State length based on order
qSize = stateLength^2;
omg = 2*pi/24;

dt = t(2) - t(1); % Get the sampling time for the subject.
[A, B, C, D] = KF.createKalmanStateSpace(order, stateLength, dt);

startT = clock;                  % Start timing the optimization.
[Cost, avgCost, Q_pop, R_pop] = KFSS.optimizeFilter(...
            iterations, mu, lambda, rho, order, lEnd, rEnd, LB, rLA, rLB,...
            A, C, t, y, f, originalSpectrum);
runTime = etime(clock, startT);  % Get the optimization runtime.

%% Use the optimized matrices in estimating the phase shift.
[~, idx] = min(Cost);

Q = reshape(Q_pop(idx, 1:qSize), stateLength, stateLength);
Q = Q*Q';
R = R_pop(idx);

% Compute P from the ARE.
[P, ~, ~] = idare(A', C', Q, R, [], []);
L = A*P*C'/(C*P*C' + R);

[xHat, yHat] = KFSS.simulateFilter(A, C, L, y);

filteredSpectrum = Utils.computeSpectrum(t, yHat);
estimCost = Utils.computeCost(originalSpectrum, filteredSpectrum, f, order);

[theta, estimPhaseShift] = Utils.estimatePhaseShift(xHat, omg, day1, day2);

fprintf("Optimization Runtime: %3.3f\n", runTime)
fprintf("Cost: %3.3f\n", estimCost)
fprintf("Phase shift: %3.4f hr\n", estimPhaseShift)

% Utils.plotAvgCostEvolution(avgCost)
% Utils.plotFilterOutputs(t, y, yHat, xHat, theta, day1, day2)