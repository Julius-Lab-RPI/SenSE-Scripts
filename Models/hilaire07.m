function [t, u, ns, y] = hilaire07(time, light, sleep_wake, initial_conditions)
% Paper:
% Addition of a non-photic component to a light-based mathematical model of
% the human circadian pacemaker - St Hilaire (2007).

% Process L Parameters.
a0 = 0.1;
p = 0.5;
beta = 0.007;
G = 37;
I0 = 9500;

% Process P Parameters.
mu = 0.13;
k = 0.55;
q = 1/3;
kc = 0.4;
tau = 24.2;

% Process Ns Parameters.
rho = 0.032;

% options = odeset("InitialStep", time(2)-time(1));
[t, y] = ode23s(@(t,y) dyn(t,y,time,light, sleep_wake),...
               time,...
               initial_conditions(:));

% Calculate u & Ns after the fact.
u = G*(a0*(light/I0).^p).*(1-y(:,3));
ns = rho*((1/3) - sleep_wake).*(1- tanh(10*y(:,1)));

    function xdot = dyn(t, x, t_rad, I_vec, sigma_vec)
        I = interp1(t_rad, I_vec, t);
        sigma = interp1(t_rad, sigma_vec, t);

        % States
        x_ = x(1);
        xc = x(2);
        n = x(3);

        xdot = zeros(3,1);

        % Process L.
        alpha = (a0*(I/I0)^p)*(I/(I+100));
        xdot(3) = 60*((alpha*(1-n)) - (beta*n));
        B_hat = G*alpha*(1-n);

        % Process Ns.
        Ns_hat = rho*((1/3) - sigma);
        Ns = Ns_hat*(1 - tanh(10*x_));

        % Process P.
        B = B_hat*(1-0.4*x_)*(1-kc*xc);
        xdot(1) = (pi/12)*(xc + mu*((x_/3)+(4*x_^3/3)-(256*x_^7/105)) + B + Ns);
        xdot(2) = (pi/12)*(q*xc*B - x_*( (24/0.99729/tau)^2 + k*B));
    end
end