function unique_filename = get_unique_filename(subject)
% Rensselaer Polytechnic Institute - Julius Lab
% SenSE Project
% Author - Chukwuemeka Osaretin Ike
%
% Description:
% Looks in the PF_Runs folder for the highest saved run number, then
% returns 1 higher to save for the subject.
folder = "PF_Runs_Delta";

if ~exist(folder, "dir")
    mkdir(folder);
end

% List all .mat files in the folder.
files = dir(fullfile(folder, strcat("A", num2str(subject), "_PF_Run_*.mat")));

run_num = length(files) + 1;

% Create the filename.
unique_filename = fullfile(folder, ...
    strcat("A", num2str(subject), "_PF_Run_", num2str(run_num), ".mat"));

end