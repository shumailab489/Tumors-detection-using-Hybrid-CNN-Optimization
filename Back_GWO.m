clc;
clear all;

% Load the data
data = readtable("D:\Research Work\BC data\data.csv");

% Convert the table to an array
dataArray = table2array(data(:, 3:end));

% Split data into features and labels
X = double(dataArray);  % Convert X to double
y = double(strcmp(data.diagnosis, 'M')); % Convert 'M' to 1 and 'B' to 0

% Data preprocessing
X = zscore(X); % Standardize input features

% Split the dataset into training and testing sets
trainRatio = 0.8;
numSamples = size(X, 1);
numTrain = round(trainRatio * numSamples);

% Shuffle the dataset before splitting
randIdx = randperm(numSamples);
X = X(randIdx, :);
y = y(randIdx);

X_train = X(1:numTrain, :);
y_train = y(1:numTrain);

X_test = X(numTrain+1:end, :);
y_test = y(numTrain+1:end);

% Neural Network parameters
N1 = 30;                % Hidden Layer Neurons
N2 = 1;                 % Output Layer Neurons 
N0 = size(X_train, 2);  % Input Layer Neurons 

% Grey Wolf Optimizer (GWO) hyperparameters
maxIter = 100; % Number of iterations
nPop = 20;    % Number of wolves

% Initialize GWO positions for weights and biases
positions = randn(nPop, N0 * N1 + N1 + N1 * N2 + N2); % Weight and bias matrix
convergenceCurve = zeros(maxIter, 1);

% Activation functions and their derivatives
tansig = @(x) tanh(x);
dtansig = @(x) 1 - tanh(x).^2;
logsig = @(x) 1./(1 + exp(-x));
dlogsig = @(x) logsig(x) .* (1 - logsig(x));

% Learning rate for backpropagation
initial_eta = 0.01;     % Initial Learning Rate
eta_decay = 0.95;       % Learning Rate Decay

% GWO process
for iter = 1:maxIter
    accuracies = zeros(nPop, 1);   
    eta = initial_eta * eta_decay^(iter-1); % Update learning rate with decay
    
    for i = 1:nPop
        % Extract weights and biases from positions
        W1 = reshape(positions(i, 1:N0*N1), [N1, N0]);
        b1 = positions(i, N0*N1+1:N0*N1+N1)';
        W2 = reshape(positions(i, N0*N1+N1+1:N0*N1+N1+N1*N2), [N2, N1]);
        b2 = positions(i, N0*N1+N1+N1*N2+1:end)';
        
        % Training the neural network with forward and backward propagation
        correctClassifications = 0;
        
        for k = 1:numTrain
            Input = X_train(k, :)';
            
            % Forward Propagation
            a1 = tansig(W1 * Input + b1);
            a2 = logsig(W2 * a1 + b2); % Logistic Sigmoidal activation function
            
            % Calculate error
            error = y_train(k) - a2;
            
            % Backward Propagation
            delta2 = error .* dlogsig(a2);
            delta1 = (W2' * delta2) .* dtansig(a1);
            
            % Gradient Descent update
            W2 = W2 + eta * delta2 * a1';
            b2 = b2 + eta * delta2;
            W1 = W1 + eta * delta1 * Input';
            b1 = b1 + eta * delta1;
            
            % Calculate accuracy
            if round(a2) == y_train(k)
                correctClassifications = correctClassifications + 1;
            end
        end
        
        % Update positions with optimized weights and biases
        positions(i, 1:N0*N1) = W1(:)';
        positions(i, N0*N1+1:N0*N1+N1) = b1';
        positions(i, N0*N1+N1+1:N0*N1+N1+N1*N2) = W2(:)';
        positions(i, N0*N1+N1+N1*N2+1:end) = b2';
        
        accuracies(i) = correctClassifications / numTrain;
    end
    
    % Update GWO positions based on fitness (accuracy)
    [~, alphaIdx] = max(accuracies);
    accuracies(alphaIdx) = -Inf; % Remove the best solution from consideration
    
    [~, betaIdx] = max(accuracies);
    accuracies(betaIdx) = -Inf;  % Remove the second best solution
    
    [~, deltaIdx] = max(accuracies);
    
    alpha = positions(alphaIdx, :);
    beta = positions(betaIdx, :);
    delta = positions(deltaIdx, :);
    
    a = 2 - iter * ((2) / maxIter);
    
    for i = 1:nPop
        for j = 1:size(positions, 2)
            r1 = rand(); % Random numbers between 0 and 1
            r2 = rand();
            A1 = 2 * a * r1 - a;
            C1 = 2 * r2;
            
            D_alpha = abs(C1 * alpha(j) - positions(i, j));
            X1 = alpha(j) - A1 * D_alpha;
            
            r1 = rand();
            r2 = rand();
            A2 = 2 * a * r1 - a;
            C2 = 2 * r2;
            
            D_beta = abs(C2 * beta(j) - positions(i, j));
            X2 = beta(j) - A2 * D_beta;
            
            r1 = rand();
            r2 = rand();
            A3 = 2 * a * r1 - a;
            C3 = 2 * r2;
            
            D_delta = abs(C3 * delta(j) - positions(i, j));
            X3 = delta(j) - A3 * D_delta;
            
            positions(i, j) = (X1 + X2 + X3) / 3;
        end
    end
    
    % Record the best accuracy for convergence analysis
    convergenceCurve(iter) = max(accuracies);
    
    % Stop if 100% accuracy is achieved
    if max(accuracies) == 1
        fprintf('100%% accuracy achieved at iteration %d\n', iter);
        break;
    end
end

% Select the best solution (position) and train neural network on the entire training set
bestPosition = positions(alphaIdx, :);
W1 = reshape(bestPosition(1:N0*N1), [N1, N0]);
b1 = bestPosition(N0*N1+1:N0*N1+N1)';
W2 = reshape(bestPosition(N0*N1+N1+1:N0*N1+N1+N1*N2), [N2, N1]);
b2 = bestPosition(N0*N1+N1+N1*N2+1:end)';

% Evaluate on training data
correctTrainClassifications = 0;
for k = 1:numTrain
    Input = X_train(k, :)';
    a1 = tansig(W1 * Input + b1);
    a2 = logsig(W2 * a1 + b2);
    
    if round(a2) == y_train(k)
        correctTrainClassifications = correctTrainClassifications + 1;
    end
end
Training_Accuracy = correctTrainClassifications * 100 / numTrain;

% Predict on test data using the final trained model
correctTestClassifications = 0;
y_pred = zeros(length(X_test), 1);
for k = 1:length(X_test)
    Input = X_test(k, :)';
    a1 = tansig(W1 * Input + b1);
    a2 = logsig(W2 * a1 + b2);
    
    y_pred(k) = round(a2);
    if y_pred(k) == y_test(k)
        correctTestClassifications = correctTestClassifications + 1;
    end
end
Testing_Accuracy = correctTestClassifications * 100 / length(X_test);

% Confusion Matrix
confMat = confusionmat(y_test, y_pred);

% Specificity, Recall, Precision
TN = confMat(1,1);
TP = confMat(2,2);
FN = confMat(2,1);
FP = confMat(1,2);

specificity = TN / (TN + FP);
recall = TP / (TP + FN);
precision = TP / (TP + FP);

% ROC Curve
[~, ~, ~, AUC] = perfcurve(y_test, y_pred, 1);

% Display final results
fprintf('Training Accuracy: %.2f%%\n', Training_Accuracy);
fprintf('Testing Accuracy: %.2f%%\n', Testing_Accuracy);
fprintf('Specificity: %.2f%%\n', specificity * 100);
fprintf('Recall: %.2f%%\n', recall * 100);
fprintf('Precision: %.2f%%\n', precision * 100);
fprintf('AUC: %.2f\n', AUC);

% Plot the convergence curve
figure;
plot(1:iter, convergenceCurve(1:iter), 'r', 'LineWidth', 1);
xlabel('Iteration');
ylabel('Accuracy');
title('Convergence Curve');
grid on;

% Plot Confusion Matrix
figure;
confusionchart(confMat);
title('Confusion Matrix');

