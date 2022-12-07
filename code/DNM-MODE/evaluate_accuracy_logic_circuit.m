function [Accuracy, Threshold_logic, K] = evaluate_accuracy_logic_circuit( input, target, pop, M, PS )

% Initialize the weights and threshold values
[G,~] = size(pop);
[I,J] = size(input);
Accuracy=zeros(1,G);
Threshold_logic=zeros(I,M,G);
K=zeros(I,M,G);

for g=1:G
    w=zeros(I,M);                  % the weight value of dendrites
    q=zeros(I,M);                  % the threshold value of dendrites

    Y=zeros(I,M,J);
    Z=ones(M,J);
    Output=zeros(1,J);
    
    Parameter=pop(g,:);
    for m=1:M
        w(:,m)= Parameter((1+2*I*(m-1)):(I+2*I*(m-1)))';
        q(:,m)= Parameter((1+I+2*I*(m-1)):(2*m*I))';
    end
    Threshold=q./w;
    
    for m=1:M
        for i=1:I
            if (0<w(i,m)&&w(i,m)<q(i,m))
                K(i,m,g)=0;
            end
            if (w(i,m)<0&&q(i,m)>0)
                K(i,m,g)=0;
            end
            if (q(i,m)<0&&w(i,m)>0)
                K(i,m,g)=2;
            end
            if (q(i,m)<w(i,m)&&w(i,m)<0)
                K(i,m,g)=2;
            end
            if (w(i,m)<q(i,m)&&q(i,m)<0)
                K(i,m,g)=-1;
            end
            if (0<q(i,m)&&q(i,m)<w(i,m))
                K(i,m,g)=1;
            end
        end
    end
    
    %% Calculate the classification rate of the logic circuit
    constant3=0;
    for j=1:J
        % build synaptic layers
        for m=1:M
            for i=1:I
                
                % Constant 1 connection
                if  K(i,m,g)==2
                    Y(i,m,j)=1;
                end
                
                % Constant 0 connection
                if  K(i,m,g)==0
                    Y(i,m,j)=0;
                end
                
                % Inverse connection
                if K(i,m,g)==-1
                    if input(i,j)<Threshold(i,m)
                        Y(i,m,j)=1;
                    else
                        Y(i,m,j)=0;
                    end
                end
                
                % Direct connection
                if K(i,m,g)==1
                    if input(i,j)<Threshold(i,m)
                        Y(i,m,j)=0;
                    else
                        Y(i,m,j)=1;
                    end
                end
            end
        end
        
        % build dendrite layers
        for m=1:M
            constant1=1;
            for i=1:I
                constant1=constant1&Y(i,m,j);
            end
            Z(m,j)=constant1;
        end
        
        % build  membrane layers
        constant2=0;
        for m=1:M
            constant2=constant2|Z(m,j);
        end
        Output(j)=constant2;
        
        if Output(j)==target(j)
            constant3=constant3+1;
        end
    end
    
    Accuracy(g) =  constant3/J;
    Threshold_logic(:,:,g) = mapminmax('reverse',Threshold, PS);
end
end


