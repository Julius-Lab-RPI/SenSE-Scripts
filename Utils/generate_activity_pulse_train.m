function [t, a] = generate_activity_pulse_train(num_days, shift_day, shift)
% Rensselaer Polytechnic Institute - Julius Lab
% SenSE Project
% Author - Chukwuemeka Osaretin Ike
%
% Description:
% Generates a simple pulse train with a shift on the specified day.

T = 24;

num_hours = num_days*T;
dt = 1/60;

t = (0:dt:num_hours)';
t_w = 2*pi/T;

% If shift_day is greater than num_days, throw an exception.
assert(shift_day < num_days, "shift_day > num_days. Cannot impart" + ...
    " a shift after the total number of days of data.")

shift_hour = shift_day*T;
shift_idx = find(t > shift_hour, 1, "first");

t_arg = [t(1:shift_idx-1); t(shift_idx:end) + shift]*t_w;

offset = 0.5;
a_max = 30;

% Currently set to 16-8 duty cycle.
a = a_max*(1 - (offset*square(t_arg, 100/3) + offset));

end