%------------------------------- Reference --------------------------------
% Ji, J., Tang, Y., Ma, L., Li, J., Lin, Q., Tang, Z., & Todo, Y. (2020). 
% Accuracy Versus Simplification in an Approximate Logic Neural Model. 
% IEEE Transactions on Neural Networks and Learning Systems.
%--------------------------------------------------------------------------
clc
clear
tic

F_index=1;                     % Problem number
popsize = 100;                 % Set the population size
Max_popsize_archive = 100;     % Set the population size of the archive
Max_iteration=300;             % Maximum iteration number
divide_rate=0.3;               % The rate of dividing the dataset


[Max1, Min2, Threshold_logic] = MODE(F_index, divide_rate, Max_iteration, popsize, Max_popsize_archive);
Max1_accuracy_train=Max1.accuracy_train;
Max1_synapse=Max1.synapse;
Max1_accuracy_test=Max1.accuracy_test;
Max1_accuracy_logic_train=Max1.accuracy_logic_train;
Max1_accuracy_logic_test=Max1.accuracy_logic_test;

Min2_synapse=Min2.synapse;
Min2_accuracy_train=Min2.accuracy_train;
Min2_accuracy_test=Min2.accuracy_test;
Min2_accuracy_logic_train=Min2.accuracy_logic_train;
Min2_accuracy_logic_test=Min2.accuracy_logic_test;

toc;
