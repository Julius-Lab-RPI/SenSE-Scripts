% Rensselaer Polytechnic Institute - Julius Lab
% SenSE Project
% Author - Chukwuemeka Osaretin Ike
%
% OBF Test Suite.
%
% Description:
% Optimizes the OBF on the specified subjects multiple times for data
% collection.

% Start the overall clock to see how long everything takes.
startAll = clock

% Optimization Hyperparameters.
iterations = 50;             % Maximum iterations.
mu = 100;                   % Number of every generation.
lambda = 50;                % Number of children.
rho = 2;                    % Number of parents to create one child.
numRuns = 40;
subjects = 10;%[3, 4, 5, 6, 7, 8, 9, 10];
orders = 1:5;%[3,4];

% Hyperparameters for creating the initial population.
LB = -5;                % Lower bound - 10^LB.
lEnd = 0;               % 
rEnd = 8;               % Twice the number of orders of magnitude.
omg = 2*pi/24;

domain = "discrete";                % "continuous" or "discrete".
if domain == "discrete"
    simulation = 'Models\Observer_Based_Filter_DT.slx';
elseif domain == "continuous"
    simulation = 'Models\Observer_Based_Filter_CT.slx';
end

% For saving the results.
saveRow = 1;
saveColumns = ["subject", "order", "run", "estimPhaseShift", "runTime", "cost"]; 
saveValues = zeros(length(subjects)*length(orders)*numRuns, length(saveColumns));
optimalParams = cell(length(subjects)*length(orders)*numRuns, 1);
avgCosts = cell(length(subjects)*length(orders)*numRuns, 1);

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

    dt = t(2) - t(1); % Get the sampling time for the subject.
    stopT = t(end);                     % Simulation stop time.

    for order = orders
        fprintf("Filter Order %d\n", order)
        stateLength = (2*order + 1);        % State length based on order
        xHatInit = zeros(stateLength, 1);    % Initial state for Simulink

    for run = 1:numRuns
        startT = clock;                  % Start timing the optimization.
        [Cost, avgCost, population] = OBF_Sim.optimizeFilter(iterations, mu, lambda,...
            rho, lEnd, rEnd, LB, order, domain, t, y, f, originalSpectrum);
        runTime = etime(clock, startT);  % Get the optimization runtime.

        % Use the optimized params in estimating the phase shift.
        [~, idx] = min(Cost);
        L = population(idx,:)';

        [A, B, C] = OBF_Sim.createStateSpace(order, dt, L, domain);

        out = sim(simulation);              % Run simulation.
        xHat = out.xHat(1:end-1,:)';        % States.
        yHat = out.yHat(1:end-1);           % Output.

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
save(strcat("Results\OBF Sim Test Suite Results A", num2str(subjects),...
    " ", num2str(numRuns), " runs.mat"), 'saveValues', 'optimalParams',...
    'avgCosts', 'saveColumns')
endAll = clock