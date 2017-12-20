%Function to train the weights and biases of the neural network according
%to the specified architecture, input data and output data. This is done
%using the Cubature Kalman Filter (to achieve second order accuracy)
function NeuralNetworkTrainingCKF(neurons, num_epochs, batch_size, eta, epsilon, qk, X_input, X_verify, Y_input, Y_verify)
%number of layers of NN
num_layers = length(neurons)-1;

%initialise the expected value vector and covariance
order = sum(neurons(1:end-1).*neurons(2:end)) + sum(neurons(2:end));
P = (1/epsilon)*eye(order);
Q = qk*eye(order);
R = (1/eta)*eye(neurons(end)*batch_size);
X = zeros(order,1);

%randomsize the weights and biases and store the indices
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
    Index(i).inp = [iws,ibs] + order;
    start = start + neurons(i)*neurons(i+1) + neurons(i+1); 
end

%augment the system with noise terms
P1 = [P, zeros(order,order), zeros(order,neurons(end)*batch_size); zeros(order,order), Q, zeros(order,neurons(end)*batch_size);...
    zeros(neurons(end)*batch_size,2*order), R];
clear P Q R;
X1 = [X;zeros(order,1);zeros(neurons(end)*batch_size,1)];
clear X;
inu = 2*order+1:2*order+neurons(end)*batch_size;

%create the weight matrices for the CKF
t_dimensionality = 2*order + neurons(end)*batch_size;
W = repmat(1/(2*t_dimensionality), [1,2*t_dimensionality]);
WeightMat = spdiags(W',0,2*t_dimensionality,2*t_dimensionality);

%define the cost function and its derivate
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
    
    %train the network using Cubature Kalman Filter
    start = 1;
    for j = 1:num_batches
        c_input = input(:,start:start+diff(j)-1);
        c_output = output(:,start:start+diff(j)-1);
        [X1,P1] = CKF(X1,P1,t_dimensionality,Index,inu,c_input,c_output,order, neurons, sigmoid,diff(j),W,WeightMat);
        start = start+diff(j);
    end

    %test it with the data meant for verification
    MSE = VerifyNN(X1, X_verify, Y_verify, sigmoid, num_layers, Index, neurons);
    fprintf('Epoc %d : %f\n', i, MSE);    
end

%CKF code
function [X,P] = CKF(X, P, n, Index, inu, input, y, order, neurons, sigmoid, batch_size, W, WeightMat)
%total number of points for 2nd order CKF
L = 2*n;
x = cubaturepoints2(X,P,n);

%prediction via unscented tranform
[X(1:order),x,P(1:order,1:order),x1] = UnscentedtransformF(x,W,WeightMat,n,L,order);          %unscented transformation of process
%update step (using uscented transform)
[Z,~,Pz,z2] = UnscentedtransformH(x,W,WeightMat,L, neurons, Index, input, sigmoid, batch_size, inu);                                       %unscented transformation of observation model
Pxy = x1*WeightMat*z2';                                         %transformed cross-covariance
K = Pxy/Pz;                                                     %kalman gain
y = reshape(y, [neurons(end)*batch_size,1]);
X(1:order) = X(1:order) + K*(y -Z);                             %state update
P(1:order,1:order) = P(1:order,1:order) - K*Pxy';               %covariance update

    
%function to generate second order cubature points
function [x,W] = cubaturepoints2(X,P,n)
%first perform cholesky decomposition to get the square root matrix
[U,D,~] = svd(P);
sqP = U*sqrt(D);
%use the same method as in sigmas2
s = sqrt(n);
temp = zeros(n, 2*n);
loc = 0:n-1;
l_index = loc*n + (loc+1);
temp(l_index) = s;
l_index = l_index + n*n;
temp(l_index) = -s;

W = repmat(1/(2*n), [1,2*n]);

Y = repmat(X, [1,2*n]);
x = Y + sqP*temp;

%function for performing unscented transform (process model)
function [Y,y,P,y1] = UnscentedtransformF(x, W,WeightMat, n, L, order)
Y = zeros(order,1);
y = zeros(n,L);
for k=1:L
    y(:,k) = x(:,k);
    %add the noise component of the process model
    y(1:order,k) = x(1:order,k) + x(order+1:2*order,k);
    %iterative computation of expected value
    Y = Y + W(k)*y(1:order,k);
end
y1 = y(1:order,:) - repmat(Y, [1,L]);
P = y1*WeightMat*y1';

%function for performing unscented transform (observation model)
function [Y,y,P,y1] = UnscentedtransformH(x, W, WeightMat, L, neurons, Index, input, sigmoid, batch_size, inu)
Y = zeros(neurons(end)*batch_size,1);
y = zeros(neurons(end)*batch_size,L);
num_layers = length(Index);
for k=1:L
    temp = input;
    for i=1:num_layers
        weights = reshape(x(Index(i).iws,k), [neurons(i+1), neurons(i)]);
        bias = x(Index(i).ibs,k);
        temp = weights*temp + repmat(bias, [1,size(temp,2)]);
        if(i~= num_layers)
            temp = sigmoid(temp);
        end
    end
    y(:,k) = reshape(temp, [neurons(end)*batch_size,1]) + x(inu,k);
    Y = Y + W(k)*y(:,k);
end
y1 = y - repmat(Y, [1,L]);
P = y1*WeightMat*y1';


%function to verify the test results
function MSE = VerifyNN(XX, X, Y, sigmoid, num_layers, Index, neurons)
temp = X;
for i = 1:num_layers
    weights = reshape(XX(Index(i).iws), [neurons(i+1), neurons(i)]);
    bias = XX(Index(i).ibs);
    temp = weights*temp + repmat(bias,[1,size(temp,2)]);
    if(i~=num_layers)
        temp = sigmoid(temp);
    end
end
MSE = mean((temp-Y).^2);
