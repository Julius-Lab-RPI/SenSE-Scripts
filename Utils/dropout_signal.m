function output = dropout_signal(signal, max_dropout_length)
% Rensselaer Polytechnic Institute - Julius Lab
% SenSE Project
% Author - Chukwuemeka Osaretin Ike
%
% Description:
% Runs through the input signal, and with some probability drops out data
% of randomly varying lengths.

output = zeros(size(signal));
idx = 1;
while idx <= length(signal)
    % Dropout with some probability. If we do drop out, draw a second
    % random number to determine how long the dropout is.
    if rand < 0.1
        idx = idx + floor(rand*max_dropout_length);
    else
        output(idx) = signal(idx);
        idx = idx + 1;
    end
end


end