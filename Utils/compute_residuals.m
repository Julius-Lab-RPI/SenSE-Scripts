function residuals = compute_residuals(true, pred)
% Rensselaer Polytechnic Institute - Julius Lab
% SenSE Project
% Author - Chukwuemeka Osaretin Ike
%
% Description:
% Computes the absolute error between a true and predicted value. Not
% really sure this needs a function, but whatever.
    true = reshape(true, size(pred));
    residuals = abs(true - pred);
end