function [t, I] = generate_daily_light(t_on, t_off, switch_high, ...
    switch_low, l_high, l_low)
% Rensselaer Polytechnic Institute - Julius Lab
% SenSE Project
% Author - Chukwuemeka Osaretin Ike
%
% Description:
% Generates a single day of light from t = [00:00 - 23:59] based on the
% method from:
% The effects of self-selected lightdark cycles and social constraints on
% human sleep and circadian timing - a modeling approach - Skeldon (2017).
%
% Need to run this sequentially to build a multi-day profile.
%
% Args:
% t_on - light on time.
% t_off - light off time.
% switch_high - time light switches from low to high level.
% switch_low - time light switches from high to low level.
% l_high - high light level.
% l_low - low light level.

% t_on = 7.5;
% t_off = 24;
% switch_high = 7.5;
% switch_low = 16.5;
% l_high = 700;
% l_low = 40;

c = 0.6;

% 24-hour time vector.
dt = 1/60;
t = 0:dt:24-dt;

% Function from Skeldon (2017).
I = l_low + ((l_high - l_low)/2)*(tanh(c*(t - switch_high)) - ...
            tanh(c*(t - switch_low)));

% Ensure zeros at the appropriate times. The raw function above only
% produces l_high and l_low values.
if t_on < t_off
    I(t <= t_on) = 0;
    I(t >= t_off) = 0;
else
    I(t >= t_off & t <= t_on) = 0;
end
end