function [r1, r2, r3]  = getindex(popsize_archive,popsize)

r1 = zeros(1, popsize);
r2 = zeros(1, popsize);
r3 = zeros(1, popsize);

for i = 1 : popsize
    rand_number1=randperm(popsize_archive);
    rand_number2=randperm(popsize);
    r1(i)=rand_number1(1);
    r2(i)=rand_number2(1);
    r3(i)=rand_number2(2);
end