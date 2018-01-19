%this function is a replica of the assimilate function courtesy of the 
%free software EnKF-Matlab. Many additional functionalities are currently
%stripped, and only the main ones are retained

%future functionalities to be included:
%1) Allow for covariance inflation at the end of the cycle with some
%predefined paramter
%2) Allow for low-frequency mean-preserving covariance rotation at the end
%3) Allow for local analysis (update one state at a time)
%4) Allow for covariance localisation (assume covariance is a band matrix
%of certain length) to avoid spurious correlation due to sampling

function [dx, A] = assimilate(n,m,var_out,A, HA, p, dy, assim_type)
%number of outputs (alternatively, can be calculated from the size of HA)
% p = length(out_loc);

%calculate some intermediate quantities
s = dy / sqrt(var_out * (m - 1));
S = HA / sqrt(var_out * (m - 1));

%calculate global perturbations in the case of EnKF
if(strcmp(assim_type, 'EnKF'))    
    D = randn(p, m) / (sqrt(m - 1));
    D_bar = mean(D,2);
    D = sqrt(m/(m-1))*(D-repmat(D_bar,[1,m]));
end

%the following is for when there is no localisation (global analysis)
if strcmp(assim_type, 'ETKF')
    M = inv(speye(m) + S' * S);
    G = M * S';
elseif m <= p
    G = inv(speye(m) + S' * S) * S';
else
    G = S' * inv(speye(p) + S * S');
end

dx = A * G * s;

switch assim_type
    case 'EnKF'
        A = A * (speye(m) + G * (D - S));
    case 'DEnKF'
        %D = 0 and half the Gain, G = 0.5*G
        A = A * (speye(m)- 0.5 * G * S);
    case 'ETKF'
        A = A * sqrtm(M);
    case 'Potter'
        T = eye(m);
        for o = 1 : p
            So = S(o, :);
            SSo = So * So' + 1;
            To = So' * (So / (SSo + sqrt(SSo)));
            S = S - S * To;
            T = T - T * To;
        end
        A = A * T;
    case 'EnOI'
        % do nothing
end
