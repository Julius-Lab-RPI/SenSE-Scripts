function [dailyPhases, thetaDReasonable] = get_average_daily_phase(t, theta, omg)
% Rensselaer Polytechnic Institute - Julius Lab
% SenSE Project
% Author - Chukwuemeka Osaretin Ike
%
% Description:
% Calculates the daily average phase value.

theta = reshape(theta, size(t));
thetaU = unwrap(theta);
thetaD = thetaU - (omg*t);
thetaDHours = thetaD/omg;
thetaDReasonable = mod(12+thetaDHours, 24) - 12;

numDays = ceil(length(t)/1440);
dailyPhases = zeros(1,numDays);

for i = 1:numDays
    dailyPhases(i) = mean(thetaDReasonable((t <= i*24) & (t >= ((i-1)*24))) );
end
end