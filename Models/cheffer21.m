function [t, y] = cheffer21(time, input, initial_conditions)
% Implementation of the Gois heart model in MATLAB.
% Paper:
% An analysis of heart rhythm dynamics using a three-coupled oscillator
% model - Gois (2009).

% Sino-Atrial parameters.
params.a_sa = 3;  
params.w_sa1 = 1;
params.w_sa2 = -1.9;
params.d_sa = 1.9;  
params.e_sa = 0.55;

% Atrio-Ventricular parameters.
params.a_av = 3;
params.w_av1 = 0.5;
params.w_av2 = -0.5;
params.d_av = 4;
params.e_av = 0.67;

% His-Purkinje complex parameters.
params.a_hp = 7;
params.w_hp1 = 1.65;
params.w_hp2 = -2;
params.d_hp = 7;
params.e_hp = 0.67;

% ECG parameters.
params.alf_0 = 1;
params.alf_1 = 0.06;
params.alf_3 = 0.1;
params.alf_5 = 0.3;

% Coupling parameters.
params.k_sa_av = 3;
params.k_sa_hp = 0;
params.k_av_sa = 0;
params.k_av_hp = 55;
params.k_hp_sa = 0;
params.k_hp_av = 0;

tau_sa_av = 0.8;
tau_av_hp = 0.1;

params.beta_t = 0.1048;

lags = [tau_sa_av, tau_av_hp];
time = time/params.beta_t; % Convert time to seconds in model's time-scale.

sol = dde23(@(t, x, Z) heart(t, x, Z, params), lags, @history, time);
t = sol.x*params.beta_t;
y = sol.y';

    function s = history(t) 
        s = [initial_conditions;
            params.alf_0 +...
            params.alf_1*initial_conditions(1) +...
            params.alf_3*initial_conditions(3) +...
            params.alf_5*initial_conditions(5)
        ];
    end

    function dx = heart(t, x, Z, params)
        x1lag = Z(1);
        x3lag = Z(3);

        % Differential equations.
        dx = zeros(7, 1);

        % SA Node.
        dx(1) = x(2);
        dx(2) = (-params.a_sa*x(2)*(x(1) - params.w_sa1)*(x(1) - params.w_sa2)) -...
            ((x(1)*(x(1) + params.d_sa)*(x(1) + params.e_sa))/(params.d_sa*params.e_sa)) -...
            (params.k_av_sa*(x(1) - x(3))) -...
            (params.k_hp_sa*(x(1) - x(5)));

        % AV Node.
        dx(3) = x(4);
        dx(4) = (-params.a_av*x(4)*(x(3) - params.w_av1)*(x(3) - params.w_av2)) -...
            ((x(3)*(x(3) + params.d_av)*(x(3) + params.e_av))/(params.d_av*params.e_av)) -...
            (params.k_sa_av*(x(3) - x1lag)) -...
            (params.k_hp_av*(x(3) - x(5)));

        % HP Complex.
        dx(5) = x(6);
        dx(6) = (-params.a_hp*x(6)*(x(5) - params.w_hp1)*(x(5) - params.w_hp2)) -...
            ((x(5)*(x(5) + params.d_hp)*(x(5) + params.e_hp))/(params.d_hp*params.e_hp)) -...
            (params.k_sa_hp*(x(5) - x(1))) -...
            (params.k_av_hp*(x(5) - x3lag));

        % ECG.
        dx(7) = (params.alf_1*x(2)) + (params.alf_3*x(4)) + (params.alf_5*x(6));
    end

end