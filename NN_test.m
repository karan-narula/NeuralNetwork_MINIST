%top-level function to read data, split the data into training sets and
%validation sets and train neural network

function [NN] = NN_test()
%control the random seed generator from here (to ensure consistent result)
seed = 0;
rng(seed);

%read the data
[X,Y, images] = ReadData();


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
batch_size = 10;
num_epochs = 10;
eta = 3.0;
neurons = [784,30,10];

%train the neural network
[NN] = NeuralNetworkTraining(neurons, num_epochs, batch_size, eta, X_train, X_test, Y_train, Y_test);

%optional, save the structure into the mat file
save('Training.mat', 'NN', 'images', 'X', 'Y');
