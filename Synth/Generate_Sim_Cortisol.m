% Rensselaer Polytechnic Institute - Julius Lab
% SenSE Project
% Author - Chukwuemeka Osaretin Ike
%
% Description:
% Generates an autoregressive noise process the same length as the activity
% profile.

%% Load activity so we have the correct length.
load("Data\Sim_Activity.mat", "t", "a")

%%

% HR uncertainty.
sigma_eps = 7;  % Std of HR.
alpha = 0.95;   % Autoregressive coefficient for HR noise process.

% Autoregressive noise component.
eps_hr = zeros(size(a));
for i = 2:length(eps_hr)
    eps_hr(i) = alpha*eps_hr(i-1) + normrnd(0, sqrt(sigma_eps));
end
eps_hr = movmean(eps_hr, 5);

%% Plots.
% figure(3); clf;
% 
% plot_light_profile(t, eps_hr)
% title("Autoregressive HR Noise Process")
% ylabel("HR (bpm)")


%% Save.
% save("Data\Sim_Cortisol", "t", "eps_hr", "sigma_eps", "alpha")