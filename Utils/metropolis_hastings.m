function [x_new, w_new] = metropolis_hastings(t, omg, x, w, y, Ns, R)
% Rensselaer Polytechnic Institute - Julius Lab
% SenSE Project
% Author - Chukwuemeka Osaretin Ike
%
% Description:
% Runs the MH resample-move algorithm to avoid particle degeneracy.
% Resamples the particles first according to their weights, then chooses
% whether to replace each particle by their resampled index based on the
% relationship between their likelihoods.

% Resample first.
[x_resampled, ~] = systematic_resample(x, w, Ns);

% Walk through each particle and make a choice based on their likelihood.
x_new = zeros(size(x));
w_new = (1/Ns)*ones(Ns, 1);
for i = 1:Ns
    ui = rand;

    % Compute likelihood of the original x.
    theta = atan2(x(i,1), x(i,2));
    phi = compute_phi(t, theta, omg);
    yHat1 = phi + x(i,4);
    circle_diff = wrapToPi(yHat1 - y);
    l1 = normpdf(circle_diff, 0, R);

    % Compute likelihood of the resampled x.
    theta = atan2(x_resampled(i,1), x_resampled(i,2));
    phi = compute_phi(t, theta, omg);
    yHat2 = phi + x_resampled(i,4);
    circle_diff = wrapToPi(yHat2 - y);
    l2 = normpdf(circle_diff, 0, R);

    if ui <= min(1, l1/l2)
        x_new(i, :) = x(i, :);
    else
        x_new(i, :) = x_resampled(i, :);
    end
end
    
end