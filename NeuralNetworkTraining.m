%Function to train the weights and biases of the neural network according
%to the specified architecture, input data and output data.

function [NN] = NeuralNetworkTraining(neurons, num_epochs, batch_size, eta, X_input, X_verify, Y_input, Y_verify)
%first initialize the data structure for storing the weights and biases in
%Neural Network
[NN,num_layers] = InitializeNN(neurons,batch_size);

%define the cost function and its derivate
sigmoid = @(x) 1./(1+exp(-x));
sigmoid_prime = @(x) sigmoid(x).*(1-sigmoid(x));


num_test = size(X_input,2);
num_batches = ceil(num_test/batch_size);
diff = [ones(num_batches-1,1)*batch_size; rem(num_test,batch_size)];
diff(diff == 0) = batch_size;

%run the training for the set number of epochs
for i =1:num_epochs
    %shuffle the data by columns (each test is in the column
    ordering = randperm(size(X_input,2));
    input = X_input(:,ordering);
    output = Y_input(:,ordering);
    %train the network using Stochatic gradient method
    NN = SGD(NN, input, output, diff,num_batches,eta, sigmoid, sigmoid_prime, num_layers);
    %test it with the data meant for verification
    test_results = VerifyNN(NN, X_verify, Y_verify, sigmoid, num_layers);
    fprintf('Epoc %d : %d/%d\n', i, test_results, length(Y_verify));    
end


%function to initialize the neural network object
function [NN, num_layers] = InitializeNN(neurons,batch_size)
num_layers = length(neurons);
NN(num_layers).biases = 0;
NN(num_layers).weights = 0;

for i = 2:num_layers
   NN(i).neurons = neurons(i);
   NN(i).bias = randn(neurons(i),1);
   NN(i).weights = randn(neurons(i),neurons(i-1));
   NN(i).z = zeros(neurons(i),1);
   NN(i).a = zeros(neurons(i),1);
end

%function to perform Stochastic gradient method
function NN = SGD(NN, X, Y, diff, num_batches, eta, sigmoid, sigmoid_prime, num_layers)
start = 1;
for i = 1:num_batches
    input = X(:,start:start+diff(i)-1);
    output = Y(:,start:start+diff(i)-1);
    for j = 1:diff(i)
       [NN] = UpdateNN(NN, input(:,j),output(:,j),eta, sigmoid, sigmoid_prime, num_layers, diff(i)); 
    end
    start = start+diff(i);
end

function [NN] = UpdateNN(NN, input,output,eta, sigmoid, sigmoid_prime, num_layers, mini_batch_length)
NN(1).a = input;
%forward pass to store the activation and and z
for i =2:num_layers
   NN(i).z = NN(i).weights*NN(i-1).a + NN(i).bias;
   NN(i).a = sigmoid(NN(i).z);
end
%backward pass to calculate the partial derivative
delta = cost_derivative(NN(end).a, output).*sigmoid_prime(NN(end).z);
NN(end).bias = NN(end).bias - (eta/mini_batch_length)*delta;
NN(end).weights = NN(end).weights - (eta/mini_batch_length)*delta*NN(end-1).a';
for i=num_layers-1:-1:2
    sp = sigmoid_prime(NN(i).z);
    delta = NN(i+1).weights'*delta.*sp;
    NN(i).bias = NN(i).bias - (eta/mini_batch_length)*delta;
    NN(i).weights = NN(i).weights - (eta/mini_batch_length)*delta*NN(i-1).a';
end


%define a function that gives derivative of a cost function
function [CD] = cost_derivative(output_a, real_output)
CD = output_a - real_output;

%function to verify the test results
function num_correct = VerifyNN(NN, X, Y, sigmoid, num_layers)
temp = X;
for i = 2:num_layers
    temp = NN(i).weights*temp + repmat(NN(i).bias,[1,size(temp,2)]);
    temp = sigmoid(temp);
end
[~, ind_nn] = max(temp);
[~, ind_ref] = max(Y);
num_correct = sum(ind_nn == ind_ref);
