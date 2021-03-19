%% Regression - part 2
%% Import and split data

clear;
close all;

data = importdata('superconductivity.csv');
data = data.data;

preproc=1;
[dataTrn,dataVal,dataChk]=split_scale(data,preproc); 
[idx,weights] = relieff(dataTrn(:,1:end-1),dataTrn(:,end),10); 

NF = [3 5 8 10];

% Radii
R = [0.3 0.5 0.8];
err = zeros(length(NF),length(R));

%% Grid Search & 5 Fold Cross Validation
for nf = 1:length(NF)
    for r = 1:length(R)
        % Split the dataset in 5 folds
        indices = crossvalind('Kfold',length(dataTrn),5);
        for k=1:5
            % Choose the best feautures
            dataReduced = [dataTrn(:,idx(1:NF(nf))) dataTrn(:,end)];
            % Choose the k-th fold to be the validation data for this round
            val =  (indices == k);
            % Training, and validation indices
            indVal = find(val == 1);
            indTrain = find(val == 0);
            
            train80 = dataReduced(indTrain,:); 
            val20 = dataReduced(indVal,:);
           
            % FIS generation options
            optFis = genfisOptions('SubtractiveClustering','ClusterInfluenceRange', R(r));
            
            % Generate fuzzy model from data
            fis = genfis(train80(:,1:end-1),train80(:,end), optFis); 

            % Training options 
            optTrain = anfisOptions;
            optTrain.InitialFIS = fis;
            optTrain.ValidationData = val20;
            optTrain.EpochNumber = 100;    

            % Train the model
            [trnFis,trnError,~,valFis,valError] = anfis(train80, optTrain);

            err(nf, r) = err(nf, r) + mean(valError);
        end
        err(nf, r) = err(nf, r) / 5;
    end
end

%% Error plots for the models
% Mean error respective to number of featuresfigure
hold on
grid on
title('Mean error respective to number of features');
xlabel('Number of features')
ylabel('RMSE')
plot(NF, err(:, 1));
plot(NF, err(:, 2));
plot(NF, err(:, 3));
legend('0.3 radius', '0.5 radius', '0.8 radius');

% Mean error respective to the cluster radius
figure
hold on
grid on
title('Mean error respective to the cluster radius');
xlabel('Cluster radius')
ylabel('RMSE')
plot(R, err(1, :));
plot(R, err(2, :));
plot(R, err(3, :));
plot(R, err(4, :));
legend('3 features', '5 features', '8 features','10 features');

%% Find the best model
[bestNF, bestNR] = find(err == min(err(:)));
disp("Best model:");
disp("     NF    NR");
disp([NF(bestNF), R(bestNR)]);
%% Final Data
train = [dataTrn(:,idx(1:NF(bestNF))) dataTrn(:,end)];
val = [dataVal(:,idx(1:NF(bestNF))) dataVal(:,end)];
chk = [dataChk(:,idx(1:NF(bestNF))) dataChk(:,end)];
radiusFinal = R(bestNR);
%% Final Model
% FIS generation options
optFisFinal = genfisOptions('SubtractiveClustering','ClusterInfluenceRange', radiusFinal);
% Generate fuzzy model from data
fisFinal = genfis(train(:,1:end-1),train(:,end),optFisFinal);
% Training options
optTrainFinal = anfisOptions('InitialFIS',fisFinal,'EpochNumber',100,'ValidationData', val);
% Train the model
[trnFisFinal,trnErrorFinal,~,valFisFinal,valErrorFinal] = anfis(train,optTrainFinal);
% Evaluate the fuzzy model
yhat = evalfis(chk(:,1:end-1),valFisFinal);
%% Plots and Errors of the final Model
%% Plot the Membership Functions of the Input Variables
for l = 1:length(trnFisFinal.input)
   figure;
   [xmf, ymf] = plotmf(fisFinal, 'input', l);
   plot(xmf, ymf);
   xlabel('Input');
   ylabel('Degree of membership');
   title(['Input #' num2str(l)]);
   figure;
   [xmf, ymf] = plotmf(trnFisFinal, 'input', l);
   plot(xmf, ymf);
   xlabel('Input');
   ylabel('Degree of membership');
   title(['Input #' num2str(l)]);
end
%% Learning Curve
figure;
hold on
title('Learning Curve');
xlabel('Epoch');
ylabel('Error');
plot(1:optTrainFinal.EpochNumber, trnErrorFinal);
hold on
plot(1:optTrainFinal.EpochNumber, valErrorFinal);
legend('Training Error', 'Validation Error');
%% Prediction Error
figure
hold on
title('Prediction Error');
xlabel('Check Dataset Sample');
ylabel('Squared Error');
plot(1:length(yhat), (yhat - chk(:,end)).^2 );
%% Plots of predictions and real values
y = chk(:,end);
figure;
hold on
title('Predictions and Real values');
xlabel('Test dataset sample')
ylabel('y')
plot(1:length(y), yhat,'Color','blue');
plot(1:length(y), y,'Color','red');
legend('Predictions', 'Real values');
%% Calculate Metrics
mean_y = mean(y);
n = length(yhat);
SS_res = sum((y-yhat).^2);
SS_tot = sum((y - mean_y).^2);
MSE = SS_res/n;
RMSE = sqrt(MSE);
Rsq = 1 - SS_res/SS_tot;
NMSE = SS_res/SS_tot;
NDEI = sqrt(NMSE); 
