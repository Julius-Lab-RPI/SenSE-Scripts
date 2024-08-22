function [t, y] = ryzhii14(time, input, initial_conditions)
% Implementation of the Ryzhii heart model in MATLAB. 
% Uses the dde23 solver, since the governing equations have 
% two delay terms.
% Ignores the muscular equations in the paper, and instead uses the
% coefficients from Gois & Savis (2009) to build the ECG.
% Paper:
% A heterogeneous coupled oscillator model for simulation
% of ECG signals - Ryzhii (2014).

% Parameters.
% Sino-Atrial.
p = struct();
p.a1 = 40;
p.u11 = 0.83;
p.u12 = -0.83;
p.d1 = 3;
p.e1 = 3.5;
p.f1 = 22;

% Atrio-Ventricular.
p.a2 = 50;
p.u21 = 0.83;
p.u22 = -0.83;
p.d2 = 3;
p.e2 = 5;
p.f2 = 8.4;
p.k_sa_av = 22;

% His-Purkinje.
p.a3 = 50;
p.u31 = 0.83;
p.u32 = -0.83;
p.d3 = 3;
p.e3 = 12;
p.f3 = 1.5;
p.k_av_hp = 22;

% ECG parameters.
p.alf_0 = 1;
p.alf_1 = 0.1;
p.alf_3 = 0.05;
p.alf_5 = 0.4;

tau_sa_av = 0.092;
tau_av_hp = 0.092;

lags = [tau_sa_av, tau_av_hp];
% tspan = [time(1), time(end)];

sol = dde23(@(t, x, Z) heart(t, x, Z, p), lags, @history, time);
t = sol.x';
y = sol.y';

% This ECG calculation and values are from Gois (2009). Ryzhii didn't
% actually give us a calculation in terms of these equations.
y(:, 7) = p.alf_0 + p.alf_1*y(:,1) + p.alf_3*y(:,3) + p.alf_5*y(:,5);

    function s = history(t) 
        % ECG initial is calculated according to the input initial
        % conditions.
        s = [initial_conditions;
            % p.alf_0 +...
            % p.alf_1*initial_conditions(1) +...
            % p.alf_3*initial_conditions(3) +...
            % p.alf_5*initial_conditions(5)
        ];
    end

    function dx = heart(t, x, Z, p)
        x2lag = Z(2);
        x4lag = Z(4);

        % Differential equations.
        dx = zeros(6, 1);

        % SA Node.
        dx(1) = x(2);
        dx(2) = (-p.a1*x(2)*(x(1) - p.u11)*(x(1) - p.u12)) -...
            (p.f1*x(1)*(x(1) + p.d1)*(x(1) + p.e1));

        % AV Node.
        dx(3) = x(4);
        dx(4) = (-p.a2*x(4)*(x(3) - p.u21)*(x(3) - p.u22)) -...
            (p.f2*x(3)*(x(3) + p.d2)*(x(3) + p.e2)) +...
            (p.k_sa_av*(x2lag - x(4)));

        % HP Complex.
        dx(5) = x(6);
        dx(6) = (-p.a3*x(6)*(x(5) - p.u31)*(x(5) - p.u32)) -...
            (p.f3*x(5)*(x(5) + p.d3)*(x(5) + p.e3)) +...
            (p.k_av_hp*(x4lag - x(6)));

        % % ECG.
        % dx(7) = (p.alf_1*x(2)) + (p.alf_3*x(4)) + (p.alf_5*x(6));
    end

end