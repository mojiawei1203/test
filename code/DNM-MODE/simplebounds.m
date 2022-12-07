% Application of simple constraints
function X=simplebounds(X,min_range, max_range)
[popsize,~]=size(X);
for i=1:popsize
    Flag4ub=X(i,:)>max_range;
    Flag4lb=X(i,:)<min_range;
    X(i,:)=(X(i,:).*(~(Flag4ub+Flag4lb)))+min_range.*Flag4lb+max_range.*Flag4ub;
end
