function [O,Error,E, w,q,w2]=D_Model_online(InputData, TeacherData, parameters)

[I,J]=size(InputData); % I=6; J=280 means that there are 280 input data in a batch.

%%%%%%%%%%%%%%%%%%%%%%%
%inialize the parameters in the model
k=parameters(1);
qs=parameters(2);
M=parameters(3); 
ita=parameters(4); % learning rate    
MaxEpoch=parameters(5);

w=2*(rand(I,M)-1/2);   
q=2*(rand(I,M)-1/2);

w2=2*(rand(M,1)-1/2);

Y=zeros(I,M,J);
Z=zeros(M,J);
V=zeros(1,J);
O=zeros(J,MaxEpoch);
E=zeros(J,MaxEpoch);
dw=zeros(I,M,J);
dq=zeros(I,M,J);
dYw=zeros(I,M,J);
dYq=zeros(I,M,J);
dZY=zeros(I,M,J);
dVZ=zeros(M,J);
dOV=zeros(1,J);
K=zeros(I,M,MaxEpoch);

dw2=zeros(M,J);%membrane parameter
dVw2=zeros(M,J);
%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%
%Adam parameters
beta1=0.9;
beta2=0.999;
ee=10e-8;

%动量和步伐
mw=zeros(I,M);%w的动量
mq=zeros(I,M);%q的动量
mw2=zeros(M,1);%w2的动量

vw=zeros(I,M);%w的步伐
vq=zeros(I,M);%q的步伐
vw2=zeros(M,1);%w2的步伐

t=1;%记录次数
% bulid Single Dendritic Neuron Model
for e=1:MaxEpoch 
 disp(['-----------Epoch: ', num2str(e), '-----------']);
    
 Sample=randperm(J);  
% build a connection layer
 for j=1:J
     for m=1:M
        for i=1:I
          if (0<w(i,m)&&w(i,m)<q(i,m))
            K(i,m,e)=0;
          end
          if (w(i,m)<0&&q(i,m)>0)
            K(i,m,e)=0;
          end
          if (q(i,m)<0&&w(i,m)>0)
            K(i,m,e)=2;
          end
          if (q(i,m)<w(i,m)&&w(i,m)<0)
            K(i,m,e)=2;
          end
          if (w(i,m)<q(i,m)&&q(i,m)<0)
            K(i,m,e)=-1;
          end
          if (0<q(i,m)&&q(i,m)<w(i,m))
            K(i,m,e)=1;
          end
          Y(i,m,Sample(j))=1/(1+exp(-k*(w(i,m)*InputData(i,Sample(j))-q(i,m))));
        end
     end

     % build a AND layer
     for m=1:M
         Q=1;
         for i=1:I
         Q=Q*Y(i,m,Sample(j));
         end
         Z(m,Sample(j))=Q;
     end
     % build a OR layer
     V=sum(w2.*Z);
%      V=sum(Z);
     % build a soma layer
     O(Sample(j),e)=1/(1+exp(-k*(V(Sample(j))-qs)));
     % compute the error
     E(Sample(j),e)=1/2*((O(Sample(j),e)-TeacherData(Sample(j)))^2);
 
     %BackPropagation-like Learning Algorithm
     for m=1:M
        for i=1:I
              dYw(i,m,Sample(j))=k*InputData(i,Sample(j))*exp(-k*(w(i,m)*InputData(i,Sample(j))-q(i,m)))/((1+exp(-k*(w(i,m)*InputData(i,Sample(j))-q(i,m))))^2);
              dYq(i,m,Sample(j))=-k*exp(-k*(w(i,m)*InputData(i,Sample(j))-q(i,m)))/((1+exp(-k*(w(i,m)*InputData(i,Sample(j))-q(i,m))))^2);
                Q=1;
                for L=1:I
                       if(L~=i)
                          Q=Q*Y(L,m,Sample(j));
                       end
                    dZY(i,m,Sample(j))=Q;
                end
              dVZ(m,Sample(j))=w2(m);
%               dVZ(m,Sample(j))=1;
              dOV(Sample(j))=k*exp(-k*(V(Sample(j))-qs))/((1+exp(-k*(V(Sample(j))-qs)))^2);
              dEO=O(Sample(j),e)-TeacherData(Sample(j));
              dw(i,m,Sample(j))=dEO*dOV(Sample(j))*dVZ(m,Sample(j))*dZY(i,m,Sample(j))*dYw(i,m,Sample(j));
              dq(i,m,Sample(j))=dEO*dOV(Sample(j))*dVZ(m,Sample(j))*dZY(i,m,Sample(j))*dYq(i,m,Sample(j));
              
              %计算当前动量和步伐
              mw(i,m) = beta1 * mw(i,m) + (1 - beta1)*dw(i,m,Sample(j));
%               vw(i,m) = beta2 * vw(i,m) + (1 - beta2)*(dw(i,m,Sample(j))^2);%w的步伐
              temp1 = beta2 * vw(i,m) + (1 - beta2)*(dw(i,m,Sample(j))^2);
              if(vw(i,m) < temp1)
                  vw(i,m)=temp1;
              end
              mq(i,m) = beta1 * mq(i,m) + (1 - beta1)*dq(i,m,Sample(j));%v的动量
%               vq(i,m) = beta2 * vq(i,m) + (1 - beta2)*(dq(i,m,Sample(j))^2);%v的步伐
              temp2 = beta2 * vq(i,m) + (1 - beta2)*(dq(i,m,Sample(j))^2);
              if(vq(i,m)<temp2)
                  vq(i,m)=temp2;
              end
              %更新权重
%               w(i,m)=w(i,m)-ita * (mw(i,m)/(1-beta1^t)) / ((vw(i,m)/(1-beta2^t))^0.5 + ee);
%               q(i,m)=q(i,m)-ita * (mq(i,m)/(1-beta1^t)) / ((vq(i,m)/(1-beta2^t))^0.5 + ee);
              w(i,m)=w(i,m)-ita * mw(i,m) / (vw(i,m)^0.5 + ee);
              q(i,m)=q(i,m)-ita * mq(i,m) / (vq(i,m)^0.5 + ee);
              %更新上一次的动量和步伐，用于下一次使用
              vw(i,m)=temp1;
              vq(i,m)=temp2;
%               w(i,m)=w(i,m)-ita*dw(i,m,Sample(j));
%               q(i,m)=q(i,m)-ita*dq(i,m,Sample(j));
        end     
            dVw2(m,Sample(j)) = Z(m,Sample(j));
            dw2(m,Sample(j)) = dEO*dOV(Sample(j))*dVw2(m,Sample(j));
            
            mw2(m) = beta1 * mw2(m) + (1-beta1) * dw2(m,Sample(j));
%             vw2(m) = beta2 * vw2(m) + (1-beta2) * (dw2(m,Sample(j))^2);
            temp = beta2 * vw2(m) + (1-beta2) * (dw2(m,Sample(j))^2);
            %temp=beta2 * vw2(m);
            if(abs(dw2(m,Sample(j)))<temp)
                vw2(m)=temp;
            end
            %更新权重
%              w2(m)=w2(m)-ita * (mw2(m)/(1-beta1^t)) / ((vw2(m)/(1-beta2^t))^0.5 + ee);
            w2(m)=w2(m) - ita * mw2(m) / (vw2(m)^0.5 + ee);
%             w2(m,1)=w2(m,1)-ita*dw2(m,Sample(j));
             vw2(m)=temp;
     end
     t=t+1;
 end
  %Show the final output
end
%Error=1/J*sum(E);%
Error=1/J*sum(E)*2;
%plot(Error);