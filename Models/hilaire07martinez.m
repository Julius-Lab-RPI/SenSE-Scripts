function [t, u, y] = hilaire07martinez(time, input, params, initial_conditions)
% Implementation of St Hilaire's model (parameters and light-sensitivity),
% but without the non-photic component. This version was used in
% Vicente-Martinez (2024).

% Paper:
% Addition of a non-photic component to a light-based mathematical model of
% the human circadian pacemaker - St Hilaire (2007).

% Modifided in:
% Uncovering personal circadian responses to light through particle swarm
% optimization - Vicente-Martinez (2024).

% This model also has tunable params.
% params = [tau; p; k].
% initial_conditions = [x0; xc0; n0].


% Process L Parameters.
a0 = 0.1;
% p = 0.5;
p = params(2);
beta = 0.007;
G = 37;
I0 = 9500;

% Process P Parameters.
mu = 0.13;
% k = 0.55;
k = params(3);
q = 1/3;
kc = 0.4;
% tau = 24.2;
tau = params(1);

% options = odeset("InitialStep", time(2)-time(1));
[t, y] = ode45(@(t,y) dyn(t,y,time,input),...
               time,...
               initial_conditions(:));

% Calculate u after the fact.
u = G*(a0*(input/I0).^p).*(1-y(:,3));

    function xdot = dyn(t, x, t_rad, I_vec)
        I = interp1(t_rad, I_vec, t);

        % States
        x_ = x(1);
        xc = x(2);
        n = x(3);

        xdot = zeros(3,1);

        % Process L.
        alpha = (a0*(I/I0)^p)*(I/(I+100));
        xdot(3) = 60*((alpha*(1-n)) - (beta*n));
        B_hat = G*alpha*(1-n);

        % Process P.
        B = B_hat*(1-0.4*x_)*(1-kc*xc);
        xdot(1) = (pi/12)*(xc + mu*((x_/3)+(4*x_^3/3)-(256*x_^7/105)) + B);
        xdot(2) = (pi/12)*(q*xc*B - x_*( (24/0.99729/tau)^2 + k*B));
    end
end