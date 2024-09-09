% Rensselaer Polytechnic Institute - Julius Lab
% SenSE Project
% Author - Chukwuemeka Osaretin Ike
%
% Observer-based Filter Matlab Optimization.
%
% Description:
% This script optimizes the OBF once on the input data.

% Start the overall clock to see how long optimization takes.
startAll = clock;

% Optimization Hyperparameters.
iterations = 50;            % Maximum iterations.
mu = 100;                   % Number of every generation.
lambda = 50;                % Number of children.
rho = 2;                    % Number of parents to create one child.
LB = -5;                % Lower bound - 10^LB.
lEnd = 0;               % 
rEnd = 8;               % Twice the number of orders of magnitude.

% **********************************************************************
% Load the data.

% % Synthetic data.
% % load('Data/jfkDataShift8.mat', 'time', 'output')
% % t = time;   y = output(:,1);
% load('Data\Mammalian_2.mat', 't', 'y')
% y = y(:,3);
% day1 = 9;           % First day to calculate phase shift
% day2 = 30;          % Second day to calculate phase shift

% Real data.
subject = 10;
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

% Take the FFT of original signal for cost calculation.
[originalSpectrum, f, len] = Utils.computeSpectrum(t, y);
originalSpectrum = originalSpectrum';

dt = t(2) - t(1); % Get the sampling time for the subject.

order = 3;
fprintf("Filter Order %d\n", order)
omg = 2*pi/24;

startT = clock;                  % Start timing the optimization.
[Cost, avgCost, population] = OBF.optimizeOBF(iterations, mu, lambda, rho,...
                        lEnd, rEnd, LB, order, t, y, f, originalSpectrum);
runTime = etime(clock, startT);  % Get the optimization runtime.

%% Use the optimized matrices in estimating the phase shift.
[~, idx] = min(Cost);
L = population(idx,:)';

[A, B, C] = OBF.createStateSpace(order, dt, L);
[xHat, yHat] = OBF.simulateFilter(t, y, A, B, C);

filteredSpectrum = Utils.computeSpectrum(t, yHat);
estimCost = Utils.computeCost(originalSpectrum, filteredSpectrum, f, order);

[theta, estimPhaseShift] = Utils.estimatePhaseShift(xHat, omg, day1, day2);

fprintf("Optimization Runtime: %3.3f\n", runTime)
fprintf("Cost: %3.3f\n", estimCost)
fprintf("Phase shift: %3.4f hr\n", estimPhaseShift)


Utils.plotAvgCostEvolution(avgCost)
Utils.plotFilterOutputs(t, y, yHat, xHat, theta, day1, day2)