% Start
function [ input_train, target_train, input_test, target_test, denNumber ] = divideDataset( F_index, divide_rate)
switch  F_index
    case 1
        input=[1 0 1 0;
               0 1 1 0];
        target=[1 1 0 0];
        denNumber=8;
    case 2
        load iris_dataset
        input=irisInputs;
        target=irisTargets(2,:);
        denNumber=8;
end

if F_index==1
    input_train=input;             input_test=input;
    target_train=target;           target_test=target;
else
    r=length(target);
    [A,~,C] = dividerand(r,divide_rate,0,1-divide_rate);
    input_train=input(:,A);         input_test=input(:,C);
    target_train=target(:,A);       target_test=target(:,C);
end
end
% Over

