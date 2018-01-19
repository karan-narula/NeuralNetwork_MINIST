%Function to train the weights and biases of the neural network according
%to the specified architecture, input data and output data. This is done
%using the Ensemble Kalman Filter for cheaper high dimensional estimation

function X = NeuralNetworkTrainingEnKF(neurons, num_epochs, batch_size, eta, epsilon, qk, num_ens, X_input, X_verify, Y_input, Y_verify)
%number of layers of NN
num_layers = length(neurons)-1;

%assimilation type (type of Ensemble filter)
assim_type = 'ETKF';

%initialise the expected value vector and covariance
order = sum(neurons(1:end-1).*neurons(2:end)) + sum(neurons(2:end));
P = (1/epsilon)*eye(order);
var_sys = qk;
var_out = (1/eta);
X = zeros(order,1);

%randomize the weights and biases and store the indices
Index(num_layers).iws = [];
start = 0;
for i =1:num_layers
    iws = start + [1:neurons(i)*neurons(i+1)];
    ibs = start + neurons(i)*neurons(i+1) + [1:neurons(i+1)];
    W = randn(neurons(i+1),neurons(i));
    b = randn(neurons(i+1),1);
    X(iws) = reshape(W, [neurons(i)*neurons(i+1),1]);
    X(ibs) = b;
    Index(i).iws = iws;
    Index(i).ibs = ibs;
    start = start + neurons(i)*neurons(i+1) + neurons(i+1); 
end

%create the ensemble for EnKF
[U,D] = svd(P);
A = sqrt(num_ens-1)*U(:,1:num_ens)*sqrt(D(1:num_ens,1:num_ens));
A = A - repmat(mean(A,2), [1,num_ens]);
%add random orthogonal rotation
V = genU(num_ens);
A = A*V;
E = A + repmat(X,[1,num_ens]);

%define the cost function and its derivative
sigmoid = @(x) 1./(1+exp(-x));


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
    
    %train the network using Ensemble Kalman Filter
    start = 1;
    for j = 1:num_batches
        c_input = input(:,start:start+diff(j)-1);
        c_output = output(:,start:start+diff(j)-1);
        %Propagate the Ensemble (prediction step)
        if(strcmp(assim_type, 'EnOI'))
            %do nothing (since only noises are added)
        else
            E = E + sqrt(var_sys)*randn(size(E));
            X = mean(E,2);
        end
        %Assimilate the Ensembles (update step)
        %need to recalculate anomalies if not EnOI
        if ~strcmp(assim_type, 'EnOI')
            X0 = mean(E,2);
            A = E - repmat(X0, [1, num_ens]);
        end
        %only this needs to be changed in case of non-linear function H
        HE = EnsembleOutputs(E, num_ens, neurons, Index, c_input, sigmoid, batch_size);
        
        %calculate prereqs
        Hx = mean(HE,2);
        dy = reshape(c_output, [neurons(end)*batch_size,1]) - Hx;
        HA = HE - repmat(Hx, [1, num_ens]);
        %assimilate function
        [dxx, A] = assimilate(order,num_ens,var_out,A, HA, length(dy), dy, assim_type);
        
        X0 = X0+dxx;
        %re-calculate ensemble from the mean
        if ~strcmp(assim_type, 'EnOI')
            E = A + repmat(X0, [1, num_ens]);
        end
        
        start = start+diff(j);
    end

    %test it with the data meant for verification
    num_correct = VerifyNN(X, X_verify, Y_verify, sigmoid, num_layers, Index, neurons);
    fprintf('Epoc %d : %d/%d\n', i, num_correct, length(Y_verify));    
end



    
%function for performing unscented transform (observation model)
function [HE] = EnsembleOutputs(E, num_ens, neurons, Index, input, sigmoid, batch_size)
HE = zeros(neurons(end)*batch_size,num_ens);
num_layers = length(Index);
for k=1:num_ens
    temp = input;
    for i=1:num_layers
        weights = reshape(E(Index(i).iws,k), [neurons(i+1), neurons(i)]);
        bias = E(Index(i).ibs,k);
        temp = weights*temp + repmat(bias, [1,size(temp,2)]);
        temp = sigmoid(temp);       
    end
    HE(:,k) = reshape(temp, [neurons(end)*batch_size,1]);
end


%function to verify the test results
function num_correct = VerifyNN(XX, X, Y, sigmoid, num_layers, Index, neurons)
temp = X;
for i = 1:num_layers
    weights = reshape(XX(Index(i).iws), [neurons(i+1), neurons(i)]);
    bias = XX(Index(i).ibs);
    temp = weights*temp + repmat(bias,[1,size(temp,2)]);
    temp = sigmoid(temp);
end
[~, ind_nn] = max(temp);
[~, ind_ref] = max(Y);
num_correct = sum(ind_nn == ind_ref);
