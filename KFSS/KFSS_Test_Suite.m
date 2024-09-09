% Rensselaer Polytechnic Institute - Julius Lab
% SenSE Project
% Author - Chukwuemeka Osaretin Ike
%
% Steady-State Kalman Filter Test Suite.
%
% Description:
% This script optimizes the SS Kalman Filter Estimator multiple times on 
% the subjects specified. It collects the data on each optimization for
% further analysis.

% Start the overall clock to see how long optimization takes.
startAll = clock

% Optimization Hyperparameters.
iterations = 50;            % Maximum iterations.
mu = 100;                   % Number of every generation.
lambda = 50;                % Number of children.
rho = 2;                    % Number of parents to create one child.
numRuns = 100;
subjects = [3, 4, 6, 7, 8, 10];         % Subjects list.

% Optimization bounds.
LB = -4;                % Q Lower bound - 10^LB.
lEnd = 0;               % Q lower end.
rEnd = 14;              % Twice the number of orders of magnitudes.
rLA = 1e2;              % R Lower bound.
rLB = 1e8;              % R upper bound.

% For saving the results.
saveRow = 1;
saveColumns = ["subject", "order", "run", "estimPhaseShift", "runTime", "cost"];
saveValues = zeros(length(subjects)*numRuns, length(saveColumns));
optimalParams = cell(length(subjects)*numRuns, 1);
avgCosts = cell(length(subjects)*numRuns, 1);

order = 1;
fprintf("Filter Order %d\n", order)
stateLength = (2*order + 1);        % State length based on order.
qSize = stateLength^2;
omg = 2*pi/24;

% Iterate through each subject specified.
for subject = subjects
    fprintf("Subject %d\n", subject)
    [t, y] = Utils.loadRealData(subject);   % Load actigraphy data.
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
    [A, B, C, D] = KF.createKalmanStateSpace(order, stateLength, dt);

    for run = 1:numRuns
        startT = clock;                  % Start timing the optimization.
        [Cost, avgCost, Q_pop, R_pop] = KFSS.optimizeFilter(...
                iterations, mu, lambda, rho, order, lEnd, rEnd, LB, rLA, rLB,...
                A, C, t, y, f, originalSpectrum);
        runTime = etime(clock, startT);  % Get the optimization runtime.

        % Use the optimized matrices in estimating the phase shift.
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

        % Add the run values to the saveValues matrix
        saveValues(saveRow,:) = [subject, order, run, estimPhaseShift,...
                                                runTime, estimCost];
        optimalParams{saveRow} = [Q_pop(idx, 1:qSize), R_pop(idx)];
        avgCosts{saveRow} = avgCost;
        saveRow = saveRow + 1;
    end
end

% Save the results.
save(strcat("Results\KFSS Test Suite Results A", num2str(subjects),...
    " ", num2str(numRuns), " runs.mat"), 'saveValues', 'optimalParams',...
    'avgCosts', 'saveColumns')
endAll = clock