function [BP] = BP_func(F_index, divide_rate, epoch, ita)
%------------------------------- Reference --------------------------------
% Ji, J., Gao, S., Cheng, J., Tang, Z., & Todo, Y. (2016). 
% An approximate logic neuron model with a dendritic structure. 
%Neurocomputing, 173, 1775-1783.
%--------------------------------------------------------------------------
[ input_train, target_train, input_test, target_test, M, PS ] = divideDataset( F_index, divide_rate);

% training the trained data
net=newnanm(input_train,target_train,ita,M,epoch);              

%% Evaluate the accuracy of ALNM
[BP.accuracy_train,BP.synapse] = evaluate_accuracy( input_train, target_train, net, M );
[BP.accuracy_test,~] = evaluate_accuracy( input_test, target_test, net, M );

%% Evaluate the accuracy of the logic circuit classifier
[BP.accuracy_logic_train, BP.Threshold_logic] = evaluate_accuracy_logic_circuit( input_train, target_train,  net, M, PS);
[BP.accuracy_logic_test, ~] = evaluate_accuracy_logic_circuit( input_test, target_test,  net, M, PS);

end
% Over


