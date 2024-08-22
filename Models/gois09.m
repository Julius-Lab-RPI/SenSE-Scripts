function [t, y] = gois09(time, input, initial_conditions)
% Implementation of the Gois heart model in MATLAB.
% Paper:
% An analysis of heart rhythm dynamics using a three-coupled oscillator
% model - Gois (2009).

% Sino-Atrial parameters.
p.a_sa = 3;  
p.w_sa1 = 0.2;
p.w_sa2 = -1.9;
p.d_sa = 3;  
p.e_sa = 4.9;

% Atrio-Ventricular parameters.
p.a_av = 3;
p.w_av1 = 0.1;
p.w_av2 = -0.1;
p.d_av = 3;
p.e_av = 3;

% His-Purkinje complex parameters.
p.a_hp = 5;
p.w_hp1 = 1;
p.w_hp2 = -1;
p.d_hp = 3;
p.e_hp = 7;

% ECG parameters.
p.alf_0 = 1;
p.alf_1 = 0.1;
p.alf_3 = 0.05;
p.alf_5 = 0.4;

% Coupling parameters.
p.k_sa_av = 0;
p.k_sa_hp = 0;
p.k_av_sa = 5;
p.k_av_hp = 0;
p.k_hp_sa = 0;
p.k_hp_av = 20;

tau_sa_av = 0.8;
tau_av_hp = 0.1;

lags = [tau_sa_av, tau_av_hp];
tspan = [time(1), time(end)];

sol = dde23(@(t, x, Z) heart(t, x, Z, p), lags, @history, tspan);
t = sol.x';
y = sol.y';

% Add a column for ECG.
y(:, 7) = p.alf_0 + p.alf_1*y(:,1) + p.alf_3*y(:,3) + p.alf_5*y(:,5);

    function s = history(t) 
        s = [initial_conditions;
            % p.alf_0 +...
            % p.alf_1*initial_conditions(1) +...
            % p.alf_3*initial_conditions(3) +...
            % p.alf_5*initial_conditions(5)
        ];
    end

    function dx = heart(t, x, Z, p)
        xlag1 = Z(1);
        xlag3 = Z(3);
        xlag5 = Z(5);

        % Differential equations.
        dx = zeros(6, 1);

        % SA Node.
        dx(1) = x(2);
        dx(2) = (-p.a_sa*x(2)*(x(1) - p.w_sa1)*(x(1) - p.w_sa2)) -...
            (x(1)*(x(1) + p.d_sa)*(x(1) + p.e_sa)) +...
            (p.k_sa_av*(x(1) - xlag3)) +...
            (p.k_sa_hp*(x(1) - xlag5));

        % AV Node.
        dx(3) = x(4);
        dx(4) = (-p.a_av*x(4)*(x(3) - p.w_av1)*(x(3) - p.w_av2)) -...
            (x(3)*(x(3) + p.d_av)*(x(3) + p.e_av)) +...
            (p.k_av_sa*(x(3) - xlag1)) +...
            (p.k_av_hp*(x(3) - xlag5));

        % HP Complex.
        dx(5) = x(6);
        dx(6) = (-p.a_hp*x(6)*(x(5) - p.w_hp1)*(x(5) - p.w_hp2)) -...
            (x(5)*(x(5) + p.d_hp)*(x(5) + p.e_hp)) +...
            (p.k_hp_sa*(x(5) - xlag1)) +...
            (p.k_hp_av*(x(5) - xlag3));

        % % ECG.
        % dx(7) = (p.alf_1*x(2)) + (p.alf_3*x(4)) + (p.alf_5*x(6));
    end

end