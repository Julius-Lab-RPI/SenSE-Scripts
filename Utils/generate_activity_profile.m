function [t, a] = generate_activity_profile(num_days, shift_day, shift, ...
    t_std, a_high_std, a_low_std, switch_std, corrupt_activity)
% Rensselaer Polytechnic Institute - Julius Lab
% SenSE Project
% Author - Chukwuemeka Osaretin Ike
%
% Basal HR Generator
%
% Description:
% Generates a realistic-looking daily activity profile.

%%

% Create a simple pulse train with desired phase shift characteristics.
[t, a_pulse] = generate_activity_pulse_train(num_days, shift_day, shift);

% a_pulse has t_on and t_off times for each day. Collect those.
t_ons = zeros(num_days, 1);
t_offs = t_ons;
for i = 1:num_days
    idx = t >= (i-1)*24 & t < i*24;
    [t_ons(i), t_offs(i)] = find_transitions(t(idx), a_pulse(idx));
end

% Perturb t_on and off for each day to generate different light on
% and off times. Also perturb the high and low activity levels for each
% day, and when the switch happens from high to low.
a = zeros(size(a_pulse));
for i = 1:num_days
    idx = t >= (i-1)*24 & t < i*24;

    % Activity levels.
    a_high = max(0, 30 + normrnd(0, a_high_std));
    a_low = max(0, 5 + normrnd(0, a_low_std));

    % Switch times from low to high and back.
    switch_high = t_ons(i) + normrnd(0, switch_std);
    switch_low = 17 + normrnd(0, switch_std);

    % Activity on and off times.
    t_on = max(0, t_ons(i) + normrnd(0, t_std));
    t_off = min(24, t_offs(i) + normrnd(0, t_std));

    % Generate the day's activity with the parameters.
    [~, a(idx)] = generate_daily_activity(t_on, t_off, switch_high, ...
            switch_low, a_high, a_low);

    % Add autoregressive noise to the day's activity.
    if corrupt_activity
        a(idx) = corrupt_activity_signal(a(idx));
    end
end


end