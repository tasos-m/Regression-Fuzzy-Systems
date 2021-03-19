%% Anastasios Mouratidis, AEM 9040
%% Regression - part 1
%% Import and split data

clear;
close all;

data = importdata('airfoil_self_noise.dat');

preproc = 1;
[dataTrn,dataVal,dataChk] = split_scale(data,preproc);
%% 4 Models, with the following number and type of membership functions

numberOfTSK = 4;
numberOfMFs = [2 3 2 3];
typeOfMFs = ["constant" "constant" "linear" "linear"];

for k=1:numberOfTSK
    
    % Display the current model
    mod = [' Model ', num2str(k)];
    disp(mod);
    

    % FIS generation options
    optFis = genfisOptions('GridPartition');
    optFis.NumMembershipFunctions = numberOfMFs(k);
    optFis.InputMembershipFunctionType = 'gbellmf';
    optFis.OutputMembershipFunctionType = typeOfMFs(k);
    
    % Generate fuzzy model from data
    modelFis = genfis(dataTrn(:, 1:end-1), dataTrn(:,end), optFis); 
    
    % Training options
    optOfTrain = anfisOptions;
    optOfTrain.InitialFIS = modelFis;
    optOfTrain.OptimizationMethod = 1;
    optOfTrain.ValidationData = dataVal;
    optOfTrain.EpochNumber = 100;
    
    % Train the model
   [trnFis,trnError,~,valFis,valError] = anfis(dataTrn, optOfTrain);
    
    % Evaluate the fuzzy model
    yhat = evalfis(dataChk(:,1:end-1), valFis);
    
    %% Plot the Membership Functions of the Input Variables
    for i = 1:length(trnFis.input) 
        figure;
        plotmf(trnFis, 'input', i); 
    end

    %% Learning curve  
    figure;
    hold on
    title(['Learning Curve, ', num2str(numberOfMFs(k)), ' member functions']);
    xlabel('Epoch');
    ylabel('Error');
    plot(1:optOfTrain.EpochNumber, trnError);
    hold on
    plot(1:optOfTrain.EpochNumber, valError);
    legend('Training Error', 'Validation Error');
    
    %% Prediction Error
    figure;
    hold on
    title(['Prediction Error ', num2str(numberOfMFs(k)), ' member functions']);
    xlabel('Check Dataset Sample');
    ylabel('Squared Error');
    plot(1:length(yhat), (yhat - dataChk(:,end)).^2 );
    
    %% Calculate Metrics
    y = dataChk(:,end);
    mean_y = mean(y);
    n = length(yhat);
    SS_res = sum((y-yhat).^2);
    SS_tot = sum((y - mean_y).^2);
    MSE = SS_res/n;
    RMSE = sqrt(MSE);
    Rsq = 1 - SS_res/SS_tot;
    NMSE = SS_res/SS_tot;
    NDEI = sqrt(NMSE);

    disp('     RMSE      NMSE      NDEI       R^2');
    disp([RMSE, NMSE, NDEI, Rsq]);
    
end