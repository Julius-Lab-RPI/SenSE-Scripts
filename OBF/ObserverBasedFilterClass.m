% Rensselaer Polytechnic Institute - Julius Lab
% Author - Chukwuemeka Osaretin Ike

% Class containing methods and properties useful for the 
% Observer-Based Filter 
classdef ObserverBasedFilterClass
   properties
        % Simulink Models
        dtSimulation = 'Models\Observer_Based_Filter_DT.slx';
        ctSimulation = 'Models\Observer_Based_Filter_CT.slx';
        
        % Filter parameters
        omg = 2*pi/24;
        zeta = 1;
        gamma_d = 1;
        gamma_omg = 0; 
        
        % Hyperparameters for creating the initial population
        LB = -5;                % Lower bound - 10^LB
        lEnd = 0;               % 
        rEnd = 8;               % Twice the number of orders of magnitudes
   end
   
   methods
        function [A, B, C, D] = createStateSpace(obj, order, dt, L, domain)
            % Creates the state-space matrices given the filter order,
            % sampling rate, and continuous-time gain vector
            
            % Returns either discrete or continuous-time matrices based 
            % on specified 'domain'

            % Number of state variables
            stateLength = (2*order + 1);
            
            % Create and populate the state matrices with the 
            % appropriate submatrices            
            Ac = zeros(stateLength); 
            Cc = zeros(1, stateLength);
            for k = 1:order
                i = k*2;
                Ac(i-1:i, i-1:i) = [0, 1; -(k*obj.omg)^2, 0];
                Cc(i) = (2*obj.zeta)/(k*obj.omg);
            end
            Cc(end) = 1;

            % Create the continous-time state space object
            A = (Ac - L*Cc);
            B = L;
            C = Cc;
            D = 0;
            contSystem = ss(A, B, C, D);

            % Discretize the system with dt sampling rate and impulse
            % method
            if domain == "discrete"
                discSystem = c2d(contSystem, dt, 'impulse');
                A = discSystem.A;
                B = discSystem.B;
                C = discSystem.C;
                D = discSystem.D;
            end
        end
        
        function population = initializePopulation(obj, mu, order)
            % Creates the initial optimization population using 
            % log sampling
            % Refer to paper for detailed explanation of this method
            
            stateLength = (2*order + 1);
            mid = (obj.rEnd+obj.lEnd)/2;
            N = (obj.rEnd-obj.lEnd)*rand(mu, stateLength) + obj.lEnd;
            
            population = zeros(size(N));
            population(N > mid+.5) = 10.^(N(N > mid+.5) - mid + obj.LB);
            population(N < mid-.5) = -10.^(mid - N(N < mid-.5) + obj.LB);
            
        end
        
        function stable = checkStability(obj, A, domain)
            % Checks if the system is stable using the appropriate method -
            % discrete or continuous
            eigenvalues = eig(A);
            
            switch domain
                case "discrete"
                    if sum(abs(eigenvalues) > 1) > 0
                        stable = false;
                    else
                        stable = true;
                    end
                case "continuous"
                    if sum((real(eigenvalues)>0) )
                        stable = false;
                    else
                        stable = true;
                    end
            end
        end
   end
    
end

