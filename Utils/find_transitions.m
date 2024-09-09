function [t_on, t_off] = find_transitions(t, I)
% Description:
% Runs through I and finds the transitions. Use on a per-day basis.
% Only works for a pulse train.

t = mod(t, 24);
t_on = 0;
t_off = 24;
for idx = 2:length(I)
    % t_on is the point where I becomes a higher value
    if I(idx) - I(idx-1) > 0
        t_on = t(idx);
    end
    if I(idx) - I(idx-1) < 0
        t_off = t(idx);
    end
end



end