% Rensselaer Polytechnic Institute - Julius Lab
% SenSE Project
% Author - Chukwuemeka Osaretin Ike
%
% Description:
% Plots the JFK x and a centered version of the basal HR to illustrate that
% their phase relationship has some elasticity.

subject = 4;


data_dir = "Data";
file_pref = strcat("Sim_A", num2str(subject));

load(fullfile(data_dir, file_pref))

mov_hr = movmean(basal_hr, 60);
cen_hr = (mov_hr - mean(mov_hr))/(std(mov_hr));
% cen_hr = (basal_hr - mean(basal_hr))/(std(basal_hr));


min_dist = 1220;
[~, x_mins] = findpeaks(-x(:,1), "MinPeakDistance", min_dist);
[~, hr_mins] = findpeaks(-cen_hr(:,1), "MinPeakDistance", min_dist);

figure(1);
plot(t, x(:,1));
hold on;
plot(t, cen_hr, "--");
xline(t(x_mins), "k-")
xline(t(hr_mins), "r--")
hold off;
xticks(0:24:t(end)); grid on;

%%
fprintf("x mins: %3.3f\n", t(x_mins))
fprintf("hr mins: %3.3f\n", t(hr_mins))