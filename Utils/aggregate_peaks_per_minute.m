function minutePeaks = aggregate_peaks_per_minute(t, y)
% Rensselaer Polytechnic Institute - Julius Lab
% SenSE Project
% Author - Chukwuemeka Osaretin Ike
%
% Description:
% Counts the number of peaks each minute. Used to convert the R point from
% ECG signal to a heart rate.

    % Make sure t starts at 0 for the for loop.
    t = t-t(1);

    % Calculate the number of full minutes in the data.
    numMinutes = floor(t(end) / 60);

    % Get all the peaks in the data, and their corresponding times
    [~, indices] = findpeaks(y, "MinPeakHeight", 0.85);
    peakTimes = t(indices);

    % Initialize an array to store the number of peaks in each minute.
    minutePeaks = zeros(numMinutes, 1);
    for i = 1:numMinutes
        % Store the number of peaks in the minutePeaks array.
        minutePeaks(i) = length(peakTimes(peakTimes >= (i-1)*60 & peakTimes < i*60));
    end
end
