function [t, y] = cheffer21_no_delay(time, input, initial_conditions)
% Implementation of the Gois heart model in MATLAB.
% Paper:
% An analysis of heart rhythm dynamics using a three-coupled oscillator
% model - Gois (2009).

% Sino-Atrial parameters.
p.a_sa = 3;  
p.w_sa1 = 1;
p.w_sa2 = -1.9;
p.d_sa = 1.9;  
p.e_sa = 0.55;

% Atrio-Ventricular parameters.
p.a_av = 3;
p.w_av1 = 0.5;
p.w_av2 = -0.5;
p.d_av = 4;
p.e_av = 0.67;

% His-Purkinje complex parameters.
p.a_hp = 7;
p.w_hp1 = 1.65;
p.w_hp2 = -2;
p.d_hp = 7;
p.e_hp = 0.67;

% ECG parameters.
p.alf_0 = 1;
p.alf_1 = 0.06;
p.alf_3 = 0.1;
p.alf_5 = 0.3;

% Coupling parameters.
p.k_sa_av = 3;
p.k_sa_hp = 0;
p.k_av_sa = 0;
p.k_av_hp = 55;
p.k_hp_sa = 0;
p.k_hp_av = 0;

% Time scaling.
p.beta_t = 0.1048;

initial_conditions = [
    initial_conditions;
    p.alf_0 +...
    p.alf_1*initial_conditions(1) +...
    p.alf_3*initial_conditions(3) +...
    p.alf_5*initial_conditions(5)
];

time = time/p.beta_t;
[t, y] = ode45(@(t, x) heart(t, x, p), [time(1) time(end)], initial_conditions(:));
t = p.beta_t*t;

    function dx = heart(t, x, p)
        % Differential equations.
        dx = zeros(7, 1);

        % SA Node.
        dx(1) = x(2);
        dx(2) = (-p.a_sa*x(2)*(x(1) - p.w_sa1)*(x(1) - p.w_sa2)) -...
            ((x(1)*(x(1) + p.d_sa)*(x(1) + p.e_sa))/(p.d_sa*p.e_sa)) -...
            (p.k_av_sa*(x(1) - x(3))) -...
            (p.k_hp_sa*(x(1) - x(5)));

        % AV Node.
        dx(3) = x(4);
        dx(4) = (-p.a_av*x(4)*(x(3) - p.w_av1)*(x(3) - p.w_av2)) -...
            ((x(3)*(x(3) + p.d_av)*(x(3) + p.e_av))/(p.d_av*p.e_av)) -...
            (p.k_sa_av*(x(3) - x(1))) -...
            (p.k_hp_av*(x(3) - x(5)));

        % HP Complex.
        dx(5) = x(6);
        dx(6) = (-p.a_hp*x(6)*(x(5) - p.w_hp1)*(x(5) - p.w_hp2)) -...
            ((x(5)*(x(5) + p.d_hp)*(x(5) + p.e_hp))/(p.d_hp*p.e_hp)) -...
            (p.k_sa_hp*(x(5) - x(1))) -...
            (p.k_av_hp*(x(5) - x(3)));

        % ECG.
        dx(7) = (p.alf_1*x(2)) + (p.alf_3*x(4)) + (p.alf_5*x(6));
    end
end