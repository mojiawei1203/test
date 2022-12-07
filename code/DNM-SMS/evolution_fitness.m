function [ Error,O] = evolution_fitness( x, T, pop, M )
%  The model can select the arbitrary axon and dendrite number
%  Every branch may have the input, N times

[II,J]=size(x);
[G,~]=size(pop);
Error=size(1,G);                % Error of each poplation
k=10;
O=zeros(J,G);

for g=1:G
    
    w=zeros(II,M);                  % the weight value of dendrites
    q=zeros(II,M);                  % the threshold value of dendrites
    
    for m=1:M
        w(:,m)= pop(g,(1+2*II*(m-1)):(II+2*II*(m-1)))';
        q(:,m)= pop(g,(1+II+2*II*(m-1)):(2*m*II))';
    end
    Y=zeros(II,M,J);
    Z=ones(M,J);
    V=zeros(1,J);
    E=zeros(1,J);
    
    %% %%%%%%%%%%%%%  dendritic neuron network %%%%%%%%%%%%%
    for j=1:J
        % build synaptic layers
        for m=1:M
            for i=1:II
                Y(i,m,j)=1/(1+exp(-k*(w(i,m)*x(i,j)-q(i,m))));
            end
        end
        
        % build dendrite layers
        for m=1:M
            Q=1;
            for i=1:II
                Q=Q*Y(i,m,j);
            end
            Z(m,j)=Q;
        end
        
        % build  membrane layers
        constant=0;
        for m=1:M
            constant=constant+Z(m,j);
        end
        V(j)=constant;
        
        % build a soma layer
        O(j,g)=1/(1+exp(-k*(V(j)-0.5)));
        E(j)=((O(j,g)-T(j))^2);
    end
    
    % calculate the error
    Error(g)=sum(E)*2/J;
end
end

