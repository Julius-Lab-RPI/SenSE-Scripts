function [t, y] = sinoatrial(time, jfk_x, initial_conditions)
% Heart model based on the Gois (2009) model, but with only the
% sino-atrial node and a coupling to the JFK model's x variable.

% Sino-Atrial parameters.
p.a1 = 40;
p.u11 = 0.83;
p.u12 = -0.83;
p.d1 = 3;
p.e1 = 3.5;
p.f1 = 22;
p.k_jfk_sa = 3.1;

time = time*3600;
[t, y] = ode45(@(t, x) heart(t, x, time, jfk_x, p),...
                time,...
                initial_conditions(:));

    function dx = heart(t, x, t_vec, jfk_vec, p)
        jfk_x_t = interp1(t_vec, jfk_vec, t);

        % Differential equations.
        dx = zeros(2, 1);

        % SA Node.
        dx(1) = x(2);
        dx(2) = (-p.a1*x(2)*(x(1) - p.u11)*(x(1) - p.u12)) -...
            (p.f1*x(1)*(x(1) + p.d1)*(x(1) + p.e1)) +...
            (p.k_jfk_sa*(jfk_x_t(2) - x(2)));
    end
end