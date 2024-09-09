function I = corrupt_daily_light(I_in, t_on, t_off)
% Rensselaer Polytechnic Institute - Julius Lab
% SenSE Project
% Author - Chukwuemeka Osaretin Ike
%
% Description:
% Corrupts a single day of light from t = [00:00 - 23:59] with an 
% auto-regressive noise process.
% Need to run this sequentially to build a multi-day profile.

% 24-hour time vector.
dt = 1/60;
% t = 0:dt:24-dt;

% Autoregressive noise component.
sigma_eps = 50;  % Std of HR.
alpha = 0.95;   % Autoregressive coefficient for HR noise process.
eps = zeros(size(I_in));
% I = zeros(size(I_in));
for i = 2:length(eps)
    if I_in(i) > 0
        eps(i) = alpha*eps(i-1) + normrnd(0, sigma_eps);
    end
end

I = max(0, I_in + eps);

% if t_on < t_off
%     I(t <= t_on) = 0;
%     I(t >= t_off) = 0;
% else
%     I(t >= t_off & t <= t_on) = 0;
% end
end