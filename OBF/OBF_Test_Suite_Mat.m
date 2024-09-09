% Rensselaer Polytechnic Institute - Julius Lab
% SenSE Project
% Author - Chukwuemeka Osaretin Ike
%
% OBF Matlab Test Suite.
%
% Description:
% Optimizes the OBF on the specified subjects multiple times for data
% collection.

% Start the overall clock to see how long everything takes.
startAll = clock

% Optimization Hyperparameters.
iterations = 50;            % Maximum iterations.
mu = 100;                   % Number of every generation.
lambda = 50;                % Number of children.
rho = 2;                    % Number of parents to create one child.
numRuns = 100;
subjects = [3, 4, 6, 7, 8, 10];
orders = [3];

% Hyperparameters for creating the initial population.
LB = -5;                % Lower bound - 10^LB.
lEnd = 0;               % 
rEnd = 8;               % Twice the number of orders of magnitude.
omg = 2*pi/24;

% For saving the results.
saveRow = 1;
saveColumns = ["subject", "order", "run", "estimPhaseShift", "runTime", "cost"]; 
saveValues = zeros(length(subjects)*length(orders)*numRuns, length(saveColumns));
optimalParams = cell(length(subjects)*length(orders)*numRuns, 1);
avgCosts = cell(length(subjects)*length(orders)*numRuns, 1);

% Iterate through each subject specified.
for subject = subjects
    fprintf("Subject %d\n", subject)
    [t, y, I, steps] = Utils.loadRealData(subject);   % Load actigraphy data.
    day1 = 7;           % First day to calculate phase shift.
    % Second day to calculate phase shift.
    if subject == 8
        day2 = 13;
    else
        day2 = 14;
    end

    % Take the FFT of original signal for cost calculations.
    [originalSpectrum, f, len] = Utils.computeSpectrum(t, y);
    originalSpectrum = originalSpectrum';

    dt = t(2) - t(1); % Get the sampling time for the subject.

    for order = orders
        fprintf("Filter Order %d\n", order)

    for run = 1:numRuns
        startT = clock;                  % Start timing the optimization.
        [Cost, avgCost, population] = OBF.optimizeOBF(iterations, mu,...
            lambda, rho, lEnd, rEnd, LB, order, t, y, f, originalSpectrum);
        runTime = etime(clock, startT);  % Get the optimization runtime.

        % Use the optimized params in estimating the phase shift.
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

        % Save the important information.
        saveValues(saveRow,:) = [subject, order, run, estimPhaseShift,...
                                                runTime, estimCost];
        optimalParams{saveRow} = L;
        avgCosts{saveRow} = avgCost;
        saveRow = saveRow + 1;
    end
    end
end

% Save the results.
save(strcat("Results\OBF Mat Test Suite Results A", num2str(subjects),...
    " ", num2str(numRuns), " runs.mat"), 'saveValues', 'optimalParams',...
    'avgCosts', 'saveColumns')
endAll = clock