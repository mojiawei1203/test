%------------------------------- Reference --------------------------------
% Ji, J., Gao, S., Cheng, J., Tang, Z., & Todo, Y. (2016). 
% An approximate logic neuron model with a dendritic structure. 
%Neurocomputing, 173, 1775-1783.
%--------------------------------------------------------------------------
clc
clear
tic

F_index=1;                     % Problem number
divide_rate=0.3;               % Train and test data rate
Max_Gen=1000;                  % Maximum iteration
ita=0.05;                      % learning rate

[BP] = BP_func(F_index, divide_rate, Max_Gen, ita);

BP_accuracy_train=BP.accuracy_train;
BP_synapse=BP.synapse;
BP_accuracy_test=BP.accuracy_test;
BP_accuracy_logic_train=BP.accuracy_logic_train;
BP_accuracy_logic_test=BP.accuracy_logic_test;

toc;
% save BP_accuracy_train BP_accuracy_train 
% save BP_synapse BP_synapse
% save BP_accuracy_test BP_accuracy_test
% save BP_accuracy_logic_train BP_accuracy_logic_train
% save BP_accuracy_logic_test BP_accuracy_logic_test
% Over