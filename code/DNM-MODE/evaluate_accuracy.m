function [accuracy, Synapse_number] = evaluate_accuracy( x, T, pop, M )

[II,J] = size(x);
[G,~] = size(pop);
k = 5;
O = zeros(J,G);
accuracy = zeros(1,G);
Synapse_number = zeros(1,G);
K=zeros(II,M,G);

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
    constant1=0;
    
    %% %%%%%%%%%%%%%  Calculation Fitness one %%%%%%%%%%%%%
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
        
        if O(j,g)>0.5
            O(j,g)=1;
        else
            O(j,g)=0;
        end
        
        if O(j,g)==T(j)
           constant1=constant1+1;
        end
    end

    accuracy(g) =  constant1/J;

    %% %%%%%%%%%%%%%  Calculation Fitness two %%%%%%%%%%%%%
    for m=1:M
        for i=1:II
            if (0<w(i,m)&&w(i,m)<q(i,m))
                K(i,m,g)=0;    % constant 0
            end
            if (w(i,m)<0&&q(i,m)>0)
                K(i,m,g)=0;    % constant 0
            end
            if (q(i,m)<0&&w(i,m)>0)
                K(i,m,g)=2;    % constant 1
            end
            if (q(i,m)<w(i,m)&&w(i,m)<0)
                K(i,m,g)=2;    % constant 1
            end
            if (w(i,m)<q(i,m)&&q(i,m)<0)
                K(i,m,g)=-1;   % Direct
            end
            if (0<q(i,m)&&q(i,m)<w(i,m))
                K(i,m,g)=1;    % Inverse
            end
        end
    end
    
    Left_synapse=II*M;
    for m=1:M
        canstant=1;  sestant=0;
        for i=1:II
            canstant=canstant*K(i,m,g);
            if K(i,m,g)==2
                sestant= sestant+1;
            end
        end
        if canstant~=0
            synapse = sestant;
        else
            synapse = II;
        end
        Left_synapse=Left_synapse-synapse;
    end

    Synapse_number(g)=Left_synapse;

end

