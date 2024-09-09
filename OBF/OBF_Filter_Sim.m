% Rensselaer Polytechnic Institute - Julius Lab
% SenSE Project
% Author - Chukwuemeka Osaretin Ike
%
% Observer-Based Filter Simulink Implementation.
%
% Description:
% Useful for testing specific gains on specific data - specify both at the
% beginning of the script.

% **********************************************************************
% Load the data.

% % Synthetic Data.
% % load('Data/jfkDataShift8SNR0.5.mat', 'time', 'output')
% load('Data/jfkDataShift8.mat', 'time', 'output')
% t = time;   y = output(:,1);
% day1 = 9;           % First day to calculate phase shift
% day2 = 25;          % Second day to calculate phase shift

% Real Data.
subject = 10;
fprintf("Subject %d\n", subject)
[t, y, I, steps] = Utils.loadRealData(subject);
% load('Data/jfkFeedSubject8.mat', 't', 'y')
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

% Previously optimized gains.
% L = [0.00130211479486675;0.0121984218501997;-0.00106193590274259;0.00393880969963419;-0.0205130372994874;0.0158846383304305;0.0311660903181032]; % Pink Noise.
% L = [-0.0385342723922105;0.0157039140563463;-3.45750203588034e-05;0.0163415412468938;-0.0105396064674146;0.0183804354652161;0.0505529238029628]; % WGN.
% L = [0.0170597500062755;0.00611129368926062;-5.24009463591830e-05;0.000825770504077374;-0.000950334147221156;0.00285882299050916;0.0204452671568324]; % Subject 3 Order 3.
% L = [-0.00622173069948799;0.00977706886746321;0.00313168508289549;0.0128872096391519;0.00112169571831373;0.00371639180576706;0.0201891980655951]; % Subject 8 Order 3.
L = [-0.00510901646364444;0.00348898713832869;-0.000777580821500233;0.000419498173323106;-0.00202647333436338;0.00608690236674181;0.0309188930283087]; % Subject 10 Order 3.

order = 3;                          % Filter order.
fprintf("Filter Order %d\n", order)
stateLength = (2*order + 1);        % State length based on order.
xHatInit = zeros(stateLength, 1);   % Initial state for Simulink.

dt = t(2) - t(1); % Get the sampling time for the subject.
stopT = t(end);                     % Simulation stop time.
domain = "discrete";                % "continuous" or "discrete".
if domain == "discrete"
    simulation = 'Models\Observer_Based_Filter_DT.slx';
elseif domain == "continuous"
    simulation = 'Models\Observer_Based_Filter_CT.slx';
end

[A, B, C] = OBF_Sim.createStateSpace(order, dt, L, domain);
out = sim(simulation);              % Run simulation.
xHat = out.xHat(1:end-1,:)';        % States.
yHat = out.yHat(1:end-1);           % Output.

filteredSpectrum = Utils.computeSpectrum(t, yHat);
estimCost = Utils.computeCost(originalSpectrum, filteredSpectrum, f, order);

omg = 2*pi/24;
[theta, phaseShift] = Utils.estimatePhaseShift(xHat, omg, day1, day2);

fprintf("Cost: %3.3f\n", estimCost)
fprintf("Phase shift: %3.4f hr\n", phaseShift)

Utils.plotFilterOutputs(t(1:end-1), y(1:end-1), yHat, xHat, theta, day1, day2)