
% parameters 
parameters=[5 0.5 5 0.0001 1000]; % k qs M ita epoch

%dataset

load sbp_39
price = sbp_;
DivideBoundary=400;
Dend = 500;
trainx= price(1:9, 1:DivideBoundary);
trainy= price(10, 1:DivideBoundary);
testx=price(1:9, DivideBoundary+1:Dend);
testy=price(10, DivideBoundary+1:Dend);

%Normalization
[trainx_normalized, st1] = mapminmax(trainx,0,1);
[trainy_normalized, st2] = mapminmax(trainy,0,1);
testx_normalized = mapminmax('apply',testx,st1);  % Make "Testing data" be normalized just the same as that with "Training data".
testy_normalized = mapminmax('apply',testy,st2);
% train
[O Error E w q w2] = D_Model_online(trainx_normalized, trainy_normalized, parameters);
trainy_predicated = mapminmax('reverse', O(:,parameters(5)), st2);

% test
 [testy_prediction_nor, Error_prediction, E_prediction]= D_Model_Prediction(testx_normalized, testy_normalized, w, q, w2, parameters);
 
 testy_prediction = mapminmax('reverse', testy_prediction_nor, st2);

% Metric
J=length(testy);
EE=((testy-testy_prediction).^2);
Mse = 1/J*sum(EE);
Mape= sum(abs((testy-testy_prediction)./testy))/length(testy);
Mae= sum(abs((testy-testy_prediction)))/length(testy);
Rmse= Mse^0.5;
yp=mean(testy);
op=mean(testy_prediction);
R= sum((testy-yp).*(testy_prediction-op))/(sum((testy-yp).*(testy-yp))*sum((testy_prediction-op).*(testy_prediction-op)))^0.5;

disp(num2str(Mse));
disp([num2str(Rmse)])
disp([num2str(Mape)])
disp([num2str(Mae)])
disp(num2str(R))


