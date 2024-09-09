function [t, I] = generate_light_profile(num_days, shift_day, shift, ...
    t_std, I_high_std, I_low_std, switch_std, corrupt_light)
% Rensselaer Polytechnic Institute - Julius Lab
% SenSE Project
% Author - Chukwuemeka Osaretin Ike
%
% Description:
% Generates a realistic light profile.
% Does so in the sequence:
%   1. Generating a simple pulse train.
%   2. Gathering the t_on and t_off times from the pulse train.
%   3. Perturbs those t_on and t_off times along with the light levels to
%       generate daily input variations.
%   4. Corrupts the light further with an autoregressive noise process.
%
% Args:
% num_days - number of days of data to generate.
% shift_day - day to introduce a phase shift in the pulse train.
% shift - number of hours to shift the pulse train on shift_day.
% t_std - daily variability in the light time on and off.
% I_max_std - daily variability in the high light levels.
% I_low_std - daily variability in the low light levels.
% switch_std - daily variability in the light level switch times.
% corrupt_light - boolean for whether to corrupt the light.

%%

% Create a simple pulse train with desired phase shift characteristics.
[t, I_pulse] = generate_light_pulse_train(num_days, shift_day, shift);

% I_pulse has t_on and t_off times for each day. Collect those.
t_ons = zeros(num_days, 1);
t_offs = t_ons;
for i = 1:num_days
    idx = t >= (i-1)*24 & t < i*24;
    [t_ons(i), t_offs(i)] = find_transitions(t(idx), I_pulse(idx));
end

% Perturb t_on and off for each day to generate different light on
% and off times. Also perturb the high and low light levels for each
% day, and when the switch happens from high to low.
I = zeros(size(I_pulse));
for i = 1:num_days
    idx = t >= (i-1)*24 & t < i*24;

    % Light levels.
    l_high = max(0, 1000 + normrnd(0, I_high_std));
    l_low = max(0, 60 + normrnd(0, I_low_std));

    % Switch times from low to high and back.
    switch_high = t_ons(i) + normrnd(0, switch_std);
    switch_low = 17 + normrnd(0, switch_std);

    % Light on and off times.
    t_on = max(0, t_ons(i) + normrnd(0, t_std));
    t_off = min(24, t_offs(i) + normrnd(0, t_std));

    % Generate the day's light with the parameters.
    [~, I(idx)] = generate_daily_light(t_on, t_off, switch_high, switch_low, ...
        l_high, l_low);

    % Add autoregressive noise to the day's light.
    if corrupt_light
        I(idx) = corrupt_daily_light(I(idx), t_on, t_off);
    end
end

end