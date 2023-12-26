classdef TSRSM < ALGORITHM
% <multi/many> <real/integer> <constrained>
% A novel tri-stage with reward-switching mechanism for constrained multiobjective optimization problems
% gr1 --- 20 --- The rewarded number of generations in the push stage
% gr2 --- 200 --- The rewarded number of generations in the pull stage
%lamda --- 1e-2 --- The threshold value for the rate of change


%------------------------------- Copyright --------------------------------
% Copyright (c) 2023 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    methods
        function main(Algorithm,Problem)

            %% Parameter setting
            [gr1,gr2,lamda] = Algorithm.ParameterSet(20,200,0.01);

            %% Generate the weight vectors
            [W,Problem.N] = UniformPoint(Problem.N,Problem.M);
            T = ceil(Problem.N/10);

            %% Detect the neighbours of each solution
            B = pdist2(W,W);
            [~,B] = sort(B,2);
            B = B(:,1:T);

            %% Generate random population
            Population1 = Problem.Initialization();  
            Fitness1    = CalFitness(Population1.objs,Population1.cons);
            Population2  = Problem.Initialization();
            Z = min(Population2.objs,[],1);
            
            %% Evaluate the Population
            Tc               = ceil(Problem.maxFE/Problem.N);
            search_stage     = 1; % 1 for push stage,otherwise,it is in pull stage.
            max_change       = 1;
            epsilon_k        = 0;
            epsilon_0        = 0;
            cp               = 2;%epsilon change rate
            alpha            = 0.95;%different epsilon update strategy
            tao              = 0.05;%the same as cp
            nr=2;
            delta=0.9;
            spop2fpr1     = zeros(ceil(Problem.maxFE/Problem.N),1);
            spop2fcd1     = zeros(ceil(Problem.maxFE/Problem.N),1);
            cnt=0;
            changerate=0.2;
            %% Optimization
            while Algorithm.NotTerminated(Population1)
                cnt =cnt+1;
                gen        = ceil(Problem.FE/(Problem.N));
                pop_cons   = Population2.cons;
                cv         = overall_cv(pop_cons);
                rf         = sum(cv <= 1e-6) / Problem.N;
                [pop2fpr1,pop2fcd1]  = Estimation(Population1.objs,1/Problem.N^(1/Problem.M));
                spop2fpr1(gen)=sum(pop2fpr1);
                spop2fcd1(gen)=sum(pop2fcd1);
                % The maximumrate of change of ideal and nadir points rk is calculated.
                if gen >= gr1
                    max_change = calc_maxchange(spop2fpr1,spop2fcd1,gen,gr1);
                end

                % The value of e(k) and the search strategy
                % are set.
                if gen < Tc
                    if max_change <= lamda && search_stage == 1
                        search_stage = 2;
                        epsilon_0 = max(cv);
                        epsilon_k = epsilon_0;
                        [fpr1,fcd1]  = Estimation(Population1.objs,1/Problem.N^(1/Problem.M));
                        finalFE_stage1=Problem.FE;
                    end
                    if search_stage == 2
                        epsilon_k =  update_epsilon(tao,epsilon_k,epsilon_0,rf,alpha,gen,Tc,cp);
                        if Problem.FE>changerate*Problem.maxFE+finalFE_stage1
                            [fpr2,fcd2]  = Estimation(Population1.objs,1/Problem.N^(1/Problem.M));

                            max_change_pop1 = calc_maxchange_pop1(sum(fpr1),sum(fpr2),sum(fcd1),sum(fcd2));
                            if max_change_pop1>lamda
                                fpr1  = fpr2;
                                fcd1 = fcd2;
                                changerate=changerate+gr2*Problem.N/Problem.maxFE;
                            else
                                 search_stage=3;
                            end
                                
                        end

                    end


                end

                % For each solution
                if search_stage==1 % The push stage
                    Offspring1  = OperatorGAhalf(Problem,Population1(randi(Problem.N,1,Problem.N)));
                    Offspring2  = OperatorGAhalf(Problem,Population2(randi(Problem.N,1,Problem.N)));
                    
                    [Population1,Fitness1] = EnvironmentalSelection([Population1,Offspring2,Offspring1],Problem.N,true);
                    [Population2,Fitness2] = EnvironmentalSelection([Population2,Offspring2,Offspring1],Problem.N,false);
                    
                elseif search_stage==2 % The pull stage
                    for i = 1 : Problem.N
                        % Choose the parents
                        if rand < delta
                            P = B(i,randperm(size(B,2)));
                        else
                            P = randperm(Problem.N);
                        end
    
                        % Generate an offspring
                        Offspring = OperatorDE(Problem,Population2(i),Population2(P(1)),Population2(P(2)));
    
                        % Update the ideal point
                        Z = min(Z,Offspring.obj);
    
                        g_old  = max(abs(Population2(P).objs-repmat(Z,length(P),1)).*W(P,:),[],2);
                        g_new  = max(repmat(abs(Offspring.obj-Z),length(P),1).*W(P,:),[],2);
                        cv_old = overall_cv(Population2(P).cons);
                        cv_new = overall_cv(Offspring.con) * ones(length(P),1);
                        % Relax constraintion
                        Population2(P(find(((g_old >= g_new) & (((cv_old <= epsilon_k) & (cv_new <= epsilon_k)) | (cv_old == cv_new)) | (cv_new < cv_old) ), nr))) = Offspring;
                        
                    end

                       MatingPool2 = TournamentSelection(2,Problem.N,Fitness1);
                       Offspring1  = OperatorGAhalf(Problem,Population1(MatingPool2));
                       [Population1,Fitness1] = EnvironmentalSelection([Population1,Population2,Offspring1],Problem.N,true);
                else% The repush stage

                        Fitness2    = CalFitness(Population2.objs);
                        MatingPool1 = TournamentSelection(2,Problem.N,Fitness2);
                        MatingPool2 = TournamentSelection(2,Problem.N,Fitness1);
                        valOffspring2 = OperatorGAhalf(Problem,Population2(MatingPool1));
                        valOffspring1 = OperatorGAhalf(Problem,Population1(MatingPool2));
                        %Knowledge transfer
                        [~,~,Next] = EnvironmentalSelection( [Population2,valOffspring2],Problem.N,true);
                        succ_rate(1,cnt) =  (sum(Next(1:Problem.N))/100) - (sum(Next(Problem.N+1:end))/50);

                        [~,~,Next] = EnvironmentalSelection( [Population1,valOffspring1],Problem.N,false);
                        succ_rate(2,cnt) =  (sum(Next(1:Problem.N))/100) - (sum(Next(Problem.N+1:end))/50);
    
                        if   succ_rate(1,cnt) >0
                            rand_number = randperm(Problem.N);
                            [Population1,Fitness1,~] = EnvironmentalSelection( [Population1,valOffspring1,Population2(rand_number(1:Problem.N/2))],Problem.N,true);
                        else
                            [Population1,Fitness1,~] = EnvironmentalSelection( [Population1,valOffspring1,valOffspring2],Problem.N,true);
                        end

                        if   succ_rate(2,cnt) >0
                            rand_number = randperm(Problem.N);
                            [Population2,Fitness2,~] = EnvironmentalSelection( [Population2,valOffspring2,Population1(rand_number(1:Problem.N/2))],Problem.N,false);
                        else
                            [Population2,Fitness2,~] = EnvironmentalSelection( [Population2,valOffspring2,valOffspring1],Problem.N,false);
                        end

                        

                end

                
            end
        end
    end
end

function result = overall_cv(cv)
% The Overall Constraint Violation
    cv(cv <= 0) = 0;cv = abs(cv);
    result = sum(cv,2);
end

function max_change = calc_maxchange(ideal_points,nadir_points,gen,last_gen)
% Calculate the Maximum Rate of Change

    delta_value = 1e-6 ;
    rz = abs((ideal_points(gen,:) - ideal_points(gen - last_gen + 1,:)) ./ max(ideal_points(gen - last_gen + 1,:),delta_value));
    nrz = abs((nadir_points(gen,:) - nadir_points(gen - last_gen + 1,:)) ./ max(nadir_points(gen - last_gen + 1,:),delta_value));
    max_change = max([rz, nrz]);
end

function result = update_epsilon(tao,epsilon_k,epsilon_0,rf,alpha,gen,Tc,cp)
    if rf < alpha
        result = (1 - tao) * epsilon_k;
    else
        result = epsilon_0 * ((1 - (gen / Tc)) ^ cp);
    end
end

function max_change = calc_maxchange_pop1(ideal_points1,ideal_points2,nadir_points1,nadir_points2)
% Calculate the Maximum Rate of Change

    delta_value = 1e-6;
    rz = abs((ideal_points1 - ideal_points2) ./ max(ideal_points1,delta_value));
    nrz = abs((nadir_points1 - nadir_points2) ./ max(nadir_points1,delta_value));
    max_change = max([rz, nrz]);
end