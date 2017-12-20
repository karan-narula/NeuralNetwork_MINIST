%another Neural Network test based on regression of data. This allows for
%small dimensionality of NN and the test for UKF-based NN training

function NN_test_regression()
%control random seed generator
seed = 0;
rng(seed);

%create artificial data on non-linear model (1D)
% sigma = 0.05;
% X_train = -10:0.2:10;
% num_train = length(X_train);
% Y_train = exp(-X_train.^2) + 0.5*exp(-(X_train-3).^2) + sigma*randn(1,num_train);
% X_test = -15:0.01:15;
% num_test = length(X_test);
% Y_test = exp(-X_test.^2) + 0.5*exp(-(X_test-3).^2) + sigma*randn(1,num_test);

%create artificial data on non-linear model (2D)
sigma = 0.05;
X_train(1,:) = -10:0.2:10;
X_train(2,:) = -5:0.1:5;
num_train = size(X_train,2);
Y_train = exp(-X_train(1,:).^2) + 0.5*exp(-(X_train(2,:)-3).^2) + sigma*randn(1,num_train);
X_test = -15:0.01:15;
X_test = [X_test;X_test];
num_test = size(X_test,2);
Y_test = exp(-X_test(1,:).^2) + 0.5*exp(-(X_test(2,:)-3).^2) + sigma*randn(1,num_test);



%dimension -> dimension_data x num_test
% X_train = X_train';
% X_test = X_test';
% Y_train = Y_train';
% Y_test = Y_test';

%define some variables for neural network training 
batch_size = 1;
num_epochs = 100;
neurons = [size(X_train,1),20,1];

%additional parameters for UKF
epsilon = 1/0.5; %to initialize P0 = (1/episolon)*I
qk = 0; %for the diagonal Q matrix; Q = qk*I
eta = 1/(0.1+sigma^2);
rng(seed);
NeuralNetworkTrainingCKF(neurons, num_epochs, batch_size, eta, epsilon, qk, X_train, X_test, Y_train, Y_test);

