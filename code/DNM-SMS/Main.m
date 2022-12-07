%------------------------------- Reference --------------------------------
% Ji, J., Song, S., Tang, Y., Gao, S., Tang, Z., & Todo, Y. (2019). 
% Approximate logic neuron model trained by states of matter search algorithm. 
% Knowledge-Based Systems, 163, 120-130.
%--------------------------------------------------------------------------
clc
clear
tic

F_index=1;                   % Problem number
divide_rate=0.3;             % Train and test data rate
popsize=50;                  % Population size
Max_Gen=1000;                % Maximum iteration

[SMS_Convergence,Percentage_train,Percentage_test] = SMS_func(F_index, divide_rate, Max_Gen, popsize);

toc;
