%top-level function to read data, split the data into training sets and
%validation sets and train neural network

function [NN] = NN_test()
%control the random seed generator from here (to ensure consistent result)
seed = 0;
rng(seed);

%filename to save the trained NN
filename = 'Training.mat';

%scale for resizing the image, scale<=1; scale = 1 -> image would be the original size
scale = 1;

%read the data if it hasn't been read before
if(exist(filename, 'file') ~=2)
    [X,Y, images] = ReadData(scale);
else
    load(filename);
end

%split the data into training set and validation set (done similar to the
%book, 50,000 and 10,000)
num_tests = 50000;
num_valid = 10000;
num_total = num_tests+num_valid;
ordering = randperm(num_total);
X_train = X(:,ordering(1:num_tests));
X_test = X(:, ordering(num_tests+1:end));
Y_train = Y(:, ordering(1:num_tests));
Y_test = Y(:, ordering(num_tests+1:end));



%define some variables for neural network training 
batch_size = 5;
num_epochs = 10;
eta = 3;
neurons = [size(X_test,1),30,10];

%train the neural network with SGD (Stochastic Gradient Descent)
rng(seed);
[NN] = NeuralNetworkTraining(neurons, num_epochs, batch_size, eta, X_train, X_test, Y_train, Y_test);

%additional parameters for Ensemble Kalman Filter (cheap enough to validate with this high dimensional example)
epsilon = 1e-1; %to initialize P0 = (1/episolon)*I
qk = 1e-4; %for the diagonal Q matrix; Q = qk*I
num_ens = 150; %number of ensemble members
rng(seed);
[NN_enkf] = NeuralNetworkTrainingEnKF(neurons, num_epochs, batch_size, eta, epsilon, qk, num_ens, X_train, X_test, Y_train, Y_test);


%optional, save the structure into the mat file
save('Training.mat', 'NN', 'NN_enkf', 'images', 'X', 'Y');
