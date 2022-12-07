function [O Error E]=D_Model_Prediction(testdata, targetdata,w,q,w2,parameters)
%
   [I,J]=size(testdata);
   k=parameters(1);
   qs=parameters(2);
   M=parameters(3); 
   Y=zeros(I,M,J);
   Z=ones(M,J);
   V=zeros(1,J);
   O=zeros(1,J);
   Q=zeros(1,J);
   E=zeros(1,J);
   output_test=zeros(1,J);
%simultion
   for j=1:J
       for m=1:M
           for i=1:I
               Y(i,m,j)=1/(1+exp(-k*(w(i,m)* testdata(i,j)-q(i,m))));
           end
       end
 % build a AND layer
      for m=1:M
          Q=1;
          for i=1:I
              Q=Q*Y(i,m,j);
          end
          Z(m,j)=Q;
      end
 % build a OR layer
%       V=sum(Z);
%        V=sum(w2.*Z-q2);
        V=sum(w2.*Z);
 % build a soma layer
      O(j)=1/(1+exp(-k*(V(j)-qs)));
      E(j)=1/2*((O(j)-targetdata(j)).^2);
   end
   Error = 1/J*sum(E)*2;
end