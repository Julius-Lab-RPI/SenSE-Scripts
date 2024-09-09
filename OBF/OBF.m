% Rensselaer Polytechnic Institute - Julius Lab
% SenSE Project
% Author - Chukwuemeka Osaretin Ike
% 
% Observer-based Filter Class.
%
% Description:
% Functions for the optimization and simulation of the OBF.
classdef OBF
    methods (Static, Access='public')
        function [A, B, C] = createStateSpace(order, dt, L, numInputs)
            % numInputs is set for the parallel Multi-Input OBF.
            % Otherwise, it's 1. Currently unused
            if ~exist('numInputs', 'var')
                numInputs = 1;
            end
            
            omg = 2*pi/24;
            
            % Continuous-time state space matrices.
            stateLength = (2*order) + 1;
            Ac = zeros(stateLength);
            Cc = zeros(1, stateLength);
            for k = 1:order
                i = k*2;
                Ac(i-1:i, i-1:i) = [0, 1; -(k*omg)^2, 0];
                Cc(i) = (2)/(k*omg);
            end
            Cc(end) = 1;

            A = (Ac - L*Cc);
            B = L;
            C = Cc;
            D = 0;
            contSystem = ss(A, B, C, D);


            % Discretize the system with dt sample time.
            discSystem = c2d(contSystem, dt, 'impulse');
            A = discSystem.A;
            B = discSystem.B;
            C = discSystem.C;
            D = discSystem.D;
        end
        
        function population = initializePopulation(mu, order, lEnd, rEnd, LB)
            % Creates the initial optimization population using 
            % log sampling
            % Refer to paper for detailed explanation of this method
            
            stateLength = (2*order + 1);
            mid = (rEnd+lEnd)/2;
            N = (rEnd-lEnd)*rand(mu, stateLength) + lEnd;
            
            population = zeros(size(N));
            population(N > mid+.5) = 10.^(N(N > mid+.5) - mid + LB);
            population(N < mid-.5) = -10.^(mid - N(N < mid-.5) + LB);
        end
        
        function [xHat, yHat] = simulateFilter(t, y, A, B, C)
            
            stateLength = size(A, 2);
            xHat = zeros(stateLength, length(t));
            xHat(:,1) = ones(stateLength, 1);

            % Run the simulation. If y=0, run without correction.
            for i = 2:length(t)
                if y(i-1) == 0
                    xHat(:,i) = A*xHat(:, i-1);
                else
                    xHat(:,i) = A*xHat(:, i-1) + B*y(i-1);
                end
            end

            % Compute the OBF output.
            yHat = C*xHat;
        end
        
        function isStable = checkStability(A)
            eigenvalues = eig(A);
            if sum(abs(eigenvalues) > 1) > 0
                isStable = false;
            else
                isStable = true;
            end 
        end
        
        function [Cost, avgCost, population] = optimizeOBF(iterations,...
                mu, lambda, rho, lEnd, rEnd, LB, order, t, y, f, originalSpectrum)
            % Arrays to hold the costs.
            Cost = zeros(mu,1);    
            avgCost = zeros(iterations, 1);
            dt = t(2) - t(1); % Get the sampling time for the subject.
            
            population = OBF.initializePopulation(mu, order, lEnd, rEnd, LB);
            
            for member = 1:mu
                % Create the state space with gain vector for current
                % member.
                L = population(member, :)';
                [A, B, C] = OBF.createStateSpace(order, dt, L);

                % Check if the system is stable.
                if ~OBF.checkStability(A)
                    Cost(member) = NaN;
                    continue
                end

                [~, yHat] = OBF.simulateFilter(t, y, A, B, C);
                filteredSpectrum = Utils.computeSpectrum(t, yHat);
                Cost(member) = Utils.computeCost(originalSpectrum, filteredSpectrum, f, order);
            end
            
            for iteration = 1:iterations
                % All possible combinations of [1:50] in pairs then take
                % first 50.
                Combination = nchoosek([1:50], rho);
                Labels = randperm(size(Combination, 1) );

                % Add 50 members to the gene pool, simulate the dynamics
                % each time and collect the costs.
                for j = 1:lambda
                    m = mu + j;

                    % Make offspring using the mean then get L.
                    population(m,:) = mean(population(Combination(...
                                                    Labels(j),:), :));
                    L = population(m,:)';
                    [A, B, C] = OBF.createStateSpace(order, dt, L);
                    if ~OBF.checkStability(A)
                        Cost(m) = NaN;
                        continue
                    end
                    [~, yHat] = OBF.simulateFilter(t, y, A, B, C);
                    filteredSpectrum = Utils.computeSpectrum(t, yHat);
                    Cost(m) = Utils.computeCost(originalSpectrum, filteredSpectrum, f, order);
                end

                % Remove the lambda highest costs - maxk only removes real values,
                % so we have to prioritize removing NaN values ourselves.
                nan_idx = find(isnan(Cost));
                [~,n] = maxk(Cost, lambda);
                remove_idx = [nan_idx; n];
                Cost(remove_idx(1:lambda)) = [];
                population(remove_idx(1:lambda), :) = [];

                % Collect the average cost of the iteration.
                avgCost(iteration) = mean(Cost);
            end
            
        end
            
    end
end