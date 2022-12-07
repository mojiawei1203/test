function [ output] = judge_domination(fitness_A, fitness_B )
% output is 1, means A<B
% output is 0, means B<A
% output is 0.5, means non-domination

D=size(fitness_A,2);
number=0;
for d=1:D
    if fitness_A(d)<fitness_B(d)
        number=number+1;
    end
end

if number==D
    output=1;
elseif number==0
    output=0;
else
    output=0.5;
end
