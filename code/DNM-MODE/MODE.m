function [Max1, Min2, Threshold_logic] = MODE(problem, divide_rate, Max_iteration, popsize, Max_popsize_archive)
%------------------------------- Reference --------------------------------
% Ji, J., Tang, Y., Ma, L., Li, J., Lin, Q., Tang, Z., & Todo, Y. (2020). 
% Accuracy Versus Simplification in an Approximate Logic Neural Model. 
% IEEE Transactions on Neural Networks and Learning Systems.
%--------------------------------------------------------------------------

F = 0.7;                      % scaling factor
CR = 0.9;                     % crossover control parameter

[input_train, target_train, input_test, target_test, denNumber] = divideDataset( problem, divide_rate);
[input_train, PS]=mapminmax(input_train,0,1);
input_test = mapminmax('apply',input_test,PS);
number_objective = 2;                                     % number_of_objectives
number_variables = 2*denNumber*size(input_train,1);       % number_of_decision_variables
min_range = -10;                                          % min_range_of_decesion_variable
max_range = 10;                                           % max_range_of_decesion_variable

%%  %%%%%%%%%%%%%%%%  Multi objective differential evolution  %%%%%%%%%%%%%%%%%%%
% Create Initial Population
population = repmat(min_range, popsize, number_variables) + rand(popsize, number_variables) .* repmat((max_range-min_range),popsize, number_variables);

% Evaluate the fitness of the population
fitness = evaluate_objective( input_train, target_train, population, denNumber, number_objective );

% Sort the initialized population
[population, fitness, rank_distance] = non_domination_sort_mod(population, fitness, number_objective, number_variables);

% Set archive population
archive=population(rank_distance(:,1)==1,:);
fitness_archive=fitness(rank_distance(:,1)==1,:);

for itera=1:Max_iteration
    %% =============     mutation       ===============
    % Get indices for mutation
    popsize_archive=size(archive,1);
    [r1, r2, r3] = getindex(popsize_archive, popsize);
    
    % Implement DE/rand/1 mutation
    population_V = archive(r1, :) + F * (population(r2, :) - population(r3, :));
    
    population_V = simplebounds(population_V, min_range, max_range);
    
    %% =============     crossover     ===============
    population_U = zeros(popsize, number_variables);
    for i = 1:popsize
        j_rand = floor(rand * number_variables) + 1;
        t = rand(1, number_variables) < CR;
        t(1, j_rand) = 1;
        t_ = 1 - t;
        population_U(i, :) = t .* population_V(i, :) + t_ .* population(i, :);
    end
    
    % Evaluate the fitness of the population_U
    fitness_U = evaluate_objective(input_train, target_train, population_U, denNumber, number_objective );
    
    %% =============       selection      ===============
    for i = 1:popsize
        output = judge_domination(fitness(i,:), fitness_U(i,:));
        if output==0 
            population(i, :) = population_U(i, :);
            fitness(i,:) = fitness_U(i,:);
            archive=[archive; population_U(i, :)];
            fitness_archive=[fitness_archive; fitness_U(i,:)];
        elseif output ==0.5
            number_dominate=0;
            number_non_dominate=0;
            for p=1:popsize_archive
                output2 = judge_domination(fitness_archive(p,:), fitness_U(i,:));
                if output2==0
                    number_dominate=number_dominate+1;
                elseif output2==0.5
                    number_non_dominate=number_non_dominate+1;
                end
            end
            if number_dominate~=0 || number_non_dominate==popsize_archive 
                population(i, :) = population_U(i, :);
                fitness(i,:) = fitness_U(i,:);
                archive=[archive; population_U(i, :)];
                fitness_archive=[fitness_archive; fitness_U(i,:)];
            end
        end
    end
    
    [archive, fitness_archive, rank_distance_archive] = non_domination_sort_mod(archive, fitness_archive, number_objective, number_variables);
    archive=archive(rank_distance_archive(:,1)==1,:);
    fitness_archive=fitness_archive(rank_distance_archive(:,1)==1,:);
    rank_distance_archive=rank_distance_archive(rank_distance_archive(:,1)==1,:);
    
    if      size(archive,1)>Max_popsize_archive
        [~, distance_Index]=sort(rank_distance_archive(:,2),'descend');
        archive=archive(distance_Index(1:Max_popsize_archive),:);
        fitness_archive=fitness_archive(distance_Index(1:Max_popsize_archive),:);
    end
end

%% Evaluate the accuracy of ALNM
[accuracy_train,synapse] = evaluate_accuracy( input_train, target_train, archive, denNumber );
[accuracy_test,~] = evaluate_accuracy( input_test, target_test, archive, denNumber );

%% Evaluate the accuracy of the logic circuit classifier
[accuracy_logic_train, Threshold_logic, ~] = evaluate_accuracy_logic_circuit( input_train, target_train, archive, denNumber, PS);
[accuracy_logic_test, ~, ~] = evaluate_accuracy_logic_circuit( input_test, target_test, archive, denNumber, PS);

%% Use the performance best 
Max1.accuracy_train=max(accuracy_train);
[Max1.accuracy_test, Max1_Index]=max(accuracy_test);
Max1.synapse=synapse(Max1_Index);
Max1.accuracy_logic_train=max(accuracy_logic_train);
Max1.accuracy_logic_test=max(accuracy_logic_test);

%% Use the simpliest structure
[Min2.synapse, Min2_Index]=min(synapse);
Min2.accuracy_train=accuracy_train(Min2_Index);
Min2.accuracy_test=accuracy_test(Min2_Index);
Min2.accuracy_logic_train=accuracy_logic_train(Min2_Index);
Min2.accuracy_logic_test=accuracy_logic_test(Min2_Index);
end
