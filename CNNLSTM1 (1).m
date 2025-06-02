clear;
clc;
seed = randi([1,1000]);
rng(seed);
% Parameters
K = 4;
M = 4;
N = 16;
Z0 = 50;
BW = 2e6;
N0_dB = -174;
sigma2_k = (10^(N0_dB/10)) * BW;
error_prob = 10^-6;
ptotal = 10^-2;
Rician_factors = 1;
total_CBL = 200;
minCBL = 20;
T = 1000;           % Total training samples
num_random_theta = 50;   % Try 50 random theta to find optimal
sigma2_k = (10^(N0_dB/10)) * BW;

%% Compute distances and path loss
BS_loc = [180, 0];               % Since RIS is at [200, 0] and BS-RIS is 20m
RIS_loc = [200, 0];              % RIS position
radius = 10;                 % Users 10 meters from RIS

% Generate random user positions on a circle around RIS
theta = 2*pi*rand(K, 1);     % Random angles
Users_loc = RIS_loc - radius * [cos(theta), sin(theta)];

% Compute distances
d_RIS_User = vecnorm(Users_loc - RIS_loc, 2, 2);      % RIS to each user
d_BS_RIS = norm(BS_loc - RIS_loc);                    % BS to RIS
d_BS_User = vecnorm(Users_loc - BS_loc, 2, 2);        % Direct path

pathLoss_dB_G = -30 - 22 * log10(d_BS_RIS);
pathLoss_dB_h = -30 - 22 * log10(d_RIS_User);
pathLoss_dB_g = -33 - 38 * log10(d_BS_User);
PL_G = 10.^(pathLoss_dB_G./10);
PL_hk = 10.^(pathLoss_dB_h./10);
PL_gk = 10.^(pathLoss_dB_g./10);


T = 10000;                      % total time steps
input_seq = zeros(N*N*2, T);    % input: Theta matrix (N x N) over time
target_seq = zeros(T, K);      % target: achievable rate per user (length-K vector)
X_data = zeros(N*M + N*K + K + K, T);      % Input
Y_data = zeros(N*N, T);            % Output (optimal Theta angles)


for t = 1:T
    % Ricean fading channels
    h_k = zeros(N, 1, K);
    for j = 1:K
        h_k(:,:,j) = sqrt(Rician_factors/(1+Rician_factors)) * sqrt(PL_hk(j)) * (randn(N, 1) + 1i * randn(N, 1)) + sqrt(1/(1+Rician_factors)) * sqrt(PL_hk(j)) * (randn(N, 1) + 1i * randn(N, 1));
    end
    
    G = sqrt(Rician_factors/(1+Rician_factors)) * sqrt(PL_G) * (randn(N, M) + 1i * randn(N, M)) + sqrt(1/(1+Rician_factors)) * sqrt(PL_G) * (randn(N, M) + 1i * randn(N, M));
    p_vec = rand(1,K);
    p_vec = p_vec/sum(p_vec);
    p_vec = p_vec.*ptotal;
    
    remaining_sum = total_CBL - K * minCBL;
    m = rand(1, K);
    m = m / sum(m);           % Normalize to sum to 1
    m = m * remaining_sum;    % Scale to desired total

    mk = m + minCBL;

    % Store input (real and imag separately)
    X_input = [real(G(:)); real(h_k(:)); p_vec(:) ; mk(:)];
    X_data(:,t) = X_input;

    best_sumrate = -inf;
    best_theta = [];
    
    % Search best Theta
    for trial = 1:num_random_theta
        
        % Generate random theta
        theta_mat = rand(N,N);
        theta_mat = (theta_mat + theta_mat') / 2;
        Theta = (1i * theta_mat + Z0 * eye(N)) \ (1i * theta_mat - Z0 * eye(N));
        theta_h = Theta(:);
        
        % Compute effective channel and rates
        initial_channel0 = zeros(K,M);
        W_init = zeros(M,K);
        H_tilda = zeros(N*N, M, K);
        for k = 1:K
            hk = h_k(:,:,k);
            Ak = zeros(N*N, N);
            hk_ext = [conj(hk); zeros((N-1)*N,1)];
            for i = 0:N-1
                Ak(:,i+1) = circshift(hk_ext, i*N);
            end
            ak = Ak*G;
            H_tilda(:,:,k) = ak;
            initial_channel0(k,:) = theta_h' * ak;
            W_init(:,k) = initial_channel0(k,:)'/norm(initial_channel0(k,:),2);
        end
        
        SINR = zeros(1,K);
        
        for k = 1:K
            interference = sum(p_vec(setdiff(1:K,k)) .* abs(initial_channel0(k,:) * W_init(:,setdiff(1:K,k))).^2);
            SINR(k) = (p_vec(k) * abs(initial_channel0(k,:) * W_init(:,k)).^2) / (sigma2_k + interference);
        end
        
        Vk = @(gamma) 1 - (1 + gamma).^(-2);
        Ck = @(gamma) log2(1 + gamma);
        
        Vk_safe = max(1e-8, Vk(SINR));
        rate = mk .* Ck(SINR) + log2(mk) - (qfuncinv(error_prob) * sqrt(mk .* Vk_safe));
        sumrate = sum(rate);
        
        % Update best
        if sumrate > best_sumrate
            best_sumrate = sumrate;
            best_theta = angle(Theta(:));   % Only store angle
        end
    end
    
    % Save optimal theta angle
    Y_data(:,t) = best_theta;
end

% Save dataset
save('ThetaOptimizerDataset.mat','X_data','Y_data');
disp('Dataset generation done!');
inputSize = N*M + N*K + K + K;
outputSize = N*N;
seqLen = 1;
layers = [
    sequenceInputLayer(inputSize,"MinLength", seqLen)
    
    convolution1dLayer(3, 128, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    
    convolution1dLayer(3, 128, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    
    flattenLayer('Name', 'flatten')

    fullyConnectedLayer(256)
    reluLayer
    %lstmLayer(256, 'OutputMode','last')
    lstmLayer(256, 'OutputMode', 'last', 'Name', 'lstm1')
    reluLayer
    dropoutLayer(0.3)

    lstmLayer(128, 'OutputMode', 'last', 'Name', 'lstm2')
    reluLayer
    dropoutLayer(0.3) 

    fullyConnectedLayer(outputSize)
    tanhLayer   % Because phase angle range ’ [-pi, pi] ’ tanh is better
    regressionLayer
];

XTrain = squeeze(num2cell(X_data, [1]));
YTrain = Y_data';
%{
options = trainingOptions('adam', ...
    'MaxEpochs', 30, ...
    'MiniBatchSize', 32, ...
    'Shuffle', 'every-epoch', ...
    'Plots', 'training-progress', ...
    'Verbose', false);
%}
options = trainingOptions('adam', ...
    'MaxEpochs', 200, ...
    'InitialLearnRate', 1e-3, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropPeriod', 10, ...
    'LearnRateDropFactor', 0.5, ...
    'MiniBatchSize', 64, ...
    'Shuffle', 'every-epoch', ...
    'Plots', 'training-progress', ...
    'Verbose', false);

net = trainNetwork(XTrain, YTrain, layers, options);

save('ThetaOptimizerTrainedModel.mat', 'net');
disp("Trained model saved successfully!");