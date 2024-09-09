function print_statistics(subject, filename, method)
% Rensselaer Polytechnic Institute - Julius Lab
% SenSE Project
% Author - Chukwuemeka Osaretin Ike
%
% Description:
% Prints the MAE and % within 1 hour of the true values for a chosen
% subject and method.


% subject = 4;

file_pref = strcat("Data\Sim_A", num2str(subject));

if method == "LCO"
    load(filename, "x")
    x_pred = x;
elseif method == "PF"
    load(filename, "x_mean")
    x_pred = x_mean;
end

% Load true data.
load(strcat(file_pref, ".mat"), "t", "x", "cbtmins")


cbtmins_pred = getCBTMins(t, x_pred(:,1), x_pred(:,2), "jewett99");

%%
idx = length(t)-1440:length(t);
fprintf("x MAE: %3.3f\n", mean(abs(x(idx,1) - x_pred(idx,1))))
fprintf("x_c MAE: %3.3f\n", mean(abs(x(idx,2) - x_pred(idx,2))))

%%
fprintf("%% within 1h: %3.3f\n", (sum(abs(cbtmins-cbtmins_pred) < 1)/length(cbtmins))*100)

end