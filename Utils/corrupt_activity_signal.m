function a = corrupt_activity_signal(a_in)
% Rensselaer Polytechnic Institute - Julius Lab
% SenSE Project
% Author - Chukwuemeka Osaretin Ike
%
% Description:
% Corrupts a single day of activity from t = [00:00 - 23:59] with an 
% auto-regressive noise process.
% Need to run this sequentially to build a multi-day profile.
%
% Args:
% t - time vector.
% a_in - activity input.
% s_2 - time when activity levels should drop to lower.

% Autoregressive noise component.
sigma_eps = 7;  % Std of HR.
alpha = 0.95;   % Autoregressive coefficient for HR noise process.
eps = zeros(size(a_in));
for i = 2:length(eps)
    if a_in(i) > 0
        eps(i) = alpha*eps(i-1) + normrnd(0, sigma_eps);
    elseif rand < 0.1
        eps(i) = max(0, normrnd(0, 2));
    end
end

a = max(0, a_in + eps);

end