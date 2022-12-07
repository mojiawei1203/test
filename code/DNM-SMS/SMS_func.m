function [SMS_Convergence,Percentage_train,Percentage_test] = SMS_func(F_index, divide_rate, Max_Gen, popsize)
%------------------------------- Reference --------------------------------
% Ji, J., Song, S., Tang, Y., Gao, S., Tang, Z., & Todo, Y. (2019). 
% Approximate logic neuron model trained by states of matter search algorithm. 
% Knowledge-Based Systems, 163, 120-130.
%--------------------------------------------------------------------------

[ input_train, target_train, input_test, target_test, denNumber ] = divideDataset( F_index, divide_rate);   
dim=2*denNumber*size(input_train,1);
down=-5;  up=5;

%paremeters
NoPob=popsize;                % Number of poblation
N_IterTotal=Max_Gen;          % Maximum iteration

phase   = 1;                  % gas phase
beta    = [0.9, 0.5, 0.1];    % movement
alpha   = [0.3, 0.05, 0];     % colides
H       = [0.9, 0.2,  0];     % random threshold
phases  = [0.5, 0.1,-0.1];    % percent of phases
param   = [0.85 0.35 0.1];    % adjustable Param

% Random initial solutions
pob = initialization(dim,NoPob,up,down);
dir = rand(NoPob,dim)*2-1;

% Eval Fitness
fitness = evolution_fitness( input_train, target_train, pob, denNumber);

% Get Best
[bestSol, bestFit] = getBest(pob,fitness, zeros(1,dim), 100000000);
SMS_Convergence=zeros(1,N_IterTotal);

for ite = 1:N_IterTotal
    
    % movement
    best = repmat(bestSol, NoPob, 1);
    b = sqrt(sum((best-pob+eps).^2,2));
    b = repmat(b,1,dim);
    a = (best-pob)./b;
    dir = dir * (1 - ite/N_IterTotal)*0.5 + a;
    
    % colis
    r = 1 * alpha(phase);
    for i = 1:NoPob - 1
        for j = i+1:NoPob
            rr = norm(pob(i,:) - pob(j,:));
            if rr < r
                c = dir(i,:);
                d = dir(j,:);
                dir(i,:)=d;
                dir(j,:)=c;
            end
        end
    end
    
    v = 1 * beta(phase) * dir;
    pob = pob + v * rand * param(phase);
    
    % random
    for i=1:NoPob
        if rand< H(phase)
            j = fix(rand*dim)+1;
            pob(i,j)= rand;
        end
    end
    
    % change of phase
    if 1 - ite/N_IterTotal < phases(phase)
        phase = phase + 1;
    end
    
    % Eval Fitness
    fitness = evolution_fitness( input_train, target_train, pob, denNumber);
    
    % Get Best
    [bestSol, bestFit] = getBest(pob,fitness, bestSol, bestFit);
    
    SMS_Convergence(ite)=min(bestFit);
    display(['The current best optimal value is : ', num2str(bestFit)]);
end

%% ------------   Simulation   ---------------------
% Verify the training dataset
[~,output_train] = evolution_fitness( input_train, target_train, bestSol, denNumber);
F=length(target_train);
Z=0;
for f=1:F
    if (output_train(f)>0.5)
        output_train(f)=1;
    else
        output_train(f)=0;
    end
    if (output_train(f)==target_train(f))
        Z=Z+1;
    end
end
fprintf('Train Percentage Correct classification   : %f%%\n', 100*Z/F);
Percentage_train=100*Z/F;

% Verify the testing dataset
[~,output_test] = evolution_fitness( input_test, target_test, bestSol, denNumber);
F=length(target_test);
Z=0;
for f=1:F
    if (output_test(f)>0.5)
        output_test(f)=1;
    else
        output_test(f)=0;
    end
    if (output_test(f)==target_test(f))
        Z=Z+1;
    end
end
fprintf('Test Percentage Correct classification   : %f%%\n', 100*Z/F);
Percentage_test=100*Z/F;
end

function [bestSol, bestFit] = getBest(pob, fitness, bestSol, bestFit)
[~, p] = min(fitness);
if fitness(p) < bestFit
    bestFit = fitness(p);
    bestSol = pob(p,:);
end
end


