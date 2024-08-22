function [t, I] = generate_light(numDays, startTime)
% Rensselaer Polytechnic Institute - Julius Lab
% SenSE Project
% Author - Chukwuemeka Osaretin Ike
%
% Description:
% Generates a light profile based on the method from the paper:
% The effects of self-selected lightdark cycles and social constraints on
% human sleep and circadian timing - a modeling approach - Skeldon (2017).

    T = 24;
    f = 1/T;
    dt = 1/60;
    numHours = numDays*T;

    % Time vector.
    t = [0+startTime:dt:numHours+startTime]';
    t_mod = mod(t, 24);

    % Light history.
    I = 40 + 330*( tanh(0.6*(t_mod-7.5)) - tanh(0.6*(t_mod-16.5)) ) + normrnd(0, 300, size(t));
    % lp_I = lowpass([I, I, I], 1/12);
    % I = lp_I(length(I)*1+1:length(I)*2);
    
    I(t_mod <= 8) = 0;
    I = max(0, I);

end