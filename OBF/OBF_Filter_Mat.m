% Rensselaer Polytechnic Institute - Julius Lab
% SenSE Project
% Author - Chukwuemeka Osaretin Ike
%
% Observer-Based Filter Matlab Implementation.
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
subject = 8;
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

% Take the FFT of original signal for cost calculation.
[originalSpectrum, f, len] = Utils.computeSpectrum(t, y);
originalSpectrum = originalSpectrum';

% Previously optimized gains.
% L = [0.0170597500062755;0.00611129368926062;-5.24009463591830e-05;0.000825770504077374;-0.000950334147221156;0.00285882299050916;0.0204452671568324]; % Subject 3 Order 3.
% L = [0.00563623223298383;0.00753628983694069;0.0284164828008145]; % Subject 7 Order 1.
% L = [0.00630865674966567;0.00801142654058331;0.0102802852598521;0.0168114861966565;0.0119769882093945]; % Subject 7 Order 2.
% L = [0.000351299075227108;0.00977139964594303;0.00160228086922235;0.0135195406269453;-0.00172461771757130;0.0222512101288383;0.0236047600359166]; % Subject 7 Order 3.
L = [-0.00370790737666416;0.00671141606241006;-0.0145620691231624;0.00373213048794183;0.0297017363743055]; % Subject 8 Order 2.
% L = [-0.00622173069948799;0.00977706886746321;0.00313168508289549;0.0128872096391519;0.00112169571831373;0.00371639180576706;0.0201891980655951]; % Subject 8 Order 3.
% L = [-0.00510901646364444;0.00348898713832869;-0.000777580821500233;0.000419498173323106;-0.00202647333436338;0.00608690236674181;0.0309188930283087]; % Subject 10 Order 3.

order = 2;                          % Filter order
fprintf("Filter Order %d\n", order)

dt = t(2) - t(1); % Get the sampling time for the subject.
[A, B, C] = OBF.createStateSpace(order, dt, L);
[xHat, yHat] = OBF.simulateFilter(t, y, A, B, C);

filteredSpectrum = Utils.computeSpectrum(t, yHat);
estimCost = Utils.computeCost(originalSpectrum, filteredSpectrum, f, order);

omg = 2*pi/24;
[theta, estimPhaseShift] = Utils.estimatePhaseShift(xHat, omg, day1, day2);

fprintf("Cost: %3.3f\n", estimCost)
fprintf("Phase shift: %3.4f hr\n", estimPhaseShift)

% Utils.plotFilterOutputs(t, y, yHat, xHat, theta, day1, day2)