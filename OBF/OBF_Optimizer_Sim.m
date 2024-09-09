% Rensselaer Polytechnic Institute - Julius Lab
% SenSE Project
% Author - Chukwuemeka Osaretin Ike
%
% Observer-based Filter Simulink Optimization.
%
% Description:
% This script optimizes the OBF once on the input data.

% Start the overall clock to see how long optimization takes
startAll = clock;

% Optimization Hyperparameters.
iterations = 15;            % Maximum iterations.
mu = 100;                   % Number of every generation.
lambda = 50;                % Number of children.
rho = 2;                    % Number of parents to create one child.
LB = -5;                % Lower bound - 10^LB.
lEnd = 0;               % 
rEnd = 8;               % Twice the number of orders of magnitude.

domain = "continuous";                % "continuous" or "discrete".
if domain == "discrete"
    simulation = 'Models\Observer_Based_Filter_DT.slx';
elseif domain == "continuous"
    simulation = 'Models\Observer_Based_Filter_CT.slx';
end

% **********************************************************************
% Load the data.

% % Synthetic data
% sToNr = 0.5;
% % load(strcat('Data/jfkDataShift8SNR', num2str(sToNr), '.mat'), 'time', 'output')
% % load('Data/jfkDataShift8PinkNoise', 'time', 'output')
% load('Data/jfkDataShift8WhiteNoise', 'time', 'output', 'noisy')
% % % load('Data/jfkDataShift8PinkNoise2Levels.mat', 'time', 'output', 'low_noisy', 'noisy')
% % load('Data/jfkDataShift8Period12.mat', 'time', 'output', 'yy')
% t = time;
% % y = output(:,1);
% y = noisy;
% % y = yy;
% day1 = 9;           % First day to calculate phase shift
% day2 = 25;          % Second day to calculate phase shift

% Real data
subject = 3;
fprintf("Subject %d\n", subject)
[t, y, I, steps] = Utils.loadRealData(subject);   % Load actigraphy data.
% load('Data/jfkFeedSubject3.mat', 't', 'y')
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

dt = t(2) - t(1); % Get the sampling time for the subject.
stopT = t(end);                     % Simulation stop time.

order = 1;
fprintf("Filter Order %d\n", order)
stateLength = (2*order + 1);        % State length based on order.
xHatInit = zeros(stateLength, 1);    % Initial state for Simulink.

startT = clock;                  % Start timing the optimization.
[Cost, avgCost, population] = OBF_Sim.optimizeFilter(iterations, mu, lambda,...
    rho, lEnd, rEnd, LB, order, domain, t, y, f, originalSpectrum);
runTime = etime(clock, startT);  % Get the optimization runtime.

%% Use the optimized params in estimating the phase shift.
[~, idx] = min(Cost);
L = population(idx,:)';

[A, B, C] = OBF_Sim.createStateSpace(order, dt, L, domain);

out = sim(simulation);              % Run simulation.
xHat = out.xHat(1:end-1,:)';        % States.
yHat = out.yHat(1:end-1);           % Output.

filteredSpectrum = Utils.computeSpectrum(t, yHat);
estimCost = Utils.computeCost(originalSpectrum, filteredSpectrum, f, order);

omg = 2*pi/24;
[theta, estimPhaseShift] = Utils.estimatePhaseShift(xHat, omg, day1, day2);

fprintf("Optimization Runtime: %3.3f\n", runTime)
fprintf("Cost: %3.3f\n", estimCost)
fprintf("Phase shift: %3.4f hr\n", estimPhaseShift)


Utils.plotAvgCostEvolution(avgCost)
Utils.plotFilterOutputs(t(1:end-1), y(1:end-1), yHat, xHat, theta, day1, day2)