% Set random seed
%seed = randi([1, 1000]);
%rng(seed);

% Initialization of variables
K = 4; M = 4; N = 16;
error_prob = 10^-8; ptotal = 10^3;
NF_dB = 3; N0_dB = 1; 
total_CBL = 100; minCBL = 10;
BW = 0.1 * 10^6; Rician_factors = 10;

% Noise power computation
sigma2_k = (10^(N0_dB/10)) * BW;
user_x = [114, 132, 148, 164];
user_y = [40, 44, 35, 45];

% RIS parameters
BS_loc = [0, 0];
RIS_loc = [40, 0];
Z0 = 50; % Impedance
L0 = 10^(-30/10);
pathloss_exp = 2.2;

% Compute distances
d_hk = sqrt((user_x - RIS_loc(1)).^2 + (user_y - RIS_loc(2)).^2);
d_g = sqrt((RIS_loc(1) - BS_loc(1))^2 + (RIS_loc(2) - BS_loc(2))^2);

% Path loss computation
pathLoss_dB_G = L0 - 10 .* pathloss_exp .* log10(d_g);
pathLoss_dB_h = L0 - 10 .* pathloss_exp .* log10(d_hk);
PL_G = 10.^(pathLoss_dB_G ./ 10);
PL_hk = 10.^(pathLoss_dB_h ./ 10);

% Ricean fading channels
h_k = zeros(N, 1, K);
for j = 1:K
    h_k(:,:,j) = sqrt(Rician_factors/(1 + Rician_factors)) * sqrt(PL_hk(j)) * (randn(N, 1) + 1i * randn(N, 1)) + ...
                  sqrt(1/(1 + Rician_factors)) * sqrt(PL_hk(j)) * (randn(N, 1) + 1i * randn(N, 1));
end

G = sqrt(Rician_factors/(1 + Rician_factors)) * sqrt(PL_G) * (randn(N, M) + 1i * randn(N, M)) + ...
    sqrt(1/(1 + Rician_factors)) * sqrt(PL_G) * (randn(N, M) + 1i * randn(N, M));

% Initialize W, Theta and CBL
W = (randn(M, K) + 1i * randn(M, K)) * 0.01;
W = W / sqrt(trace(W * W') / ptotal); % Normalize transmit power

theta = rand(N, N) * 0.01;
theta = (theta + theta') / 2;
Theta = (1i * theta + Z0 * eye(N)) \ (1i * theta - Z0 * eye(N));
theta_h = Theta(:);

initial_channel0 = zeros(K, M);
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
end

initial_interference = zeros(K, K);
for k_dash = 1:K
    initial_interference(:, k_dash) = initial_channel0 * W(:, k_dash);
end

% Initialize SINR, Blocklength and Adam optimizer variables
SINR_single_connected = zeros(1, K);
ck = max((total_CBL) .* rand(1, K), minCBL);
learningRate = 0.00005;
beta1 = 0.9; beta2 = 0.999;
epsilon = 1e-8; numEpochs = 1000;
grad_clip_threshold = 0.05;

velocityW = zeros(size(W));
velocityTheta = zeros(size(theta));
velocityCBL = zeros(size(ck));
squaredGradientW = zeros(size(W));
squaredGradientTheta = zeros(size(theta));
squaredGradientCBL = zeros(size(ck));
t = 1;
reward_history = zeros(3, numEpochs);

train_data = [];
train_labels = [];

% Optimization Loop
for epoch = 1:numEpochs
    % Update SINR
    SINR_single_connected = zeros(1, K);
    for k = 1:K
        interference_power = sum(abs(initial_channel0(k,:) * W(:, setdiff(1:K, k))).^2);
        SINR_single_connected(k) = abs(initial_channel0(k,:) * W(:, k)).^2 / (sigma2_k + interference_power);
    end
    
    % Reward calculation with Rate formula
    Vk = @(gamma) 1 - (1 + gamma).^(-2);
    Ck = @(gamma) log2(1 + gamma);
    Rate_Single_Connected = ck .* Ck(SINR_single_connected) + log2(ck) - (qfuncinv(error_prob) * sqrt(ck .* Vk(SINR_single_connected)));
    reward0 = sum(Rate_Single_Connected);
    
    % Adam optimizer updates (W, Theta, CBL)
    gradientsW = randn(size(W)) * 0.01; 
    [W, velocityW, squaredGradientW] = adamOptimizer(W, gradientsW, velocityW, squaredGradientW, learningRate, beta1, beta2, epsilon, t, grad_clip_threshold);
    
    gradientsTheta = randn(size(theta)) * 0.01;
    [theta, velocityTheta, squaredGradientTheta] = adamOptimizer(theta, gradientsTheta, velocityTheta, squaredGradientTheta, learningRate, beta1, beta2, epsilon, t, grad_clip_threshold);
    
    gradientsCBL = randn(size(ck)) * 0.01; 
    [ck, velocityCBL, squaredGradientCBL] = adamOptimizer(ck, gradientsCBL, velocityCBL, squaredGradientCBL, learningRate, beta1, beta2, epsilon, t, grad_clip_threshold);
    
    % Collect training data for Neural Network
    state1 = [abs(initial_interference(:))', angle(initial_interference(:))'];
    state2 = [ck, vecnorm(initial_channel0, 2), vecnorm(W, 2), angle(initial_channel0(:))', angle(H_tilda(:))', angle(W(:))'];
    state3 = theta(:)';
    initial_state = [state1, state2, state3, reward0]';
    
    train_data = [train_data; initial_state'];
    train_labels = [train_labels; reward0];

    reward_history(1, epoch) = reward0;
    reward_history(2, epoch) = reward0;
    reward_history(3, epoch) = reward0;
    
    disp(['Epoch ' num2str(epoch) ': Reward = ' num2str(reward0)]);
    t = t + 1;
end

% Neural Network layers and training
layers = [
    featureInputLayer(size(train_data, 2), 'Normalization', 'zscore')
    fullyConnectedLayer(256)
    reluLayer
    dropoutLayer(0.2)
    fullyConnectedLayer(128)
    reluLayer
    dropoutLayer(0.2)
    fullyConnectedLayer(64)
    reluLayer
    fullyConnectedLayer(1)
    regressionLayer
];

options = trainingOptions('adam', ...
    'MaxEpochs', 200, ...
    'MiniBatchSize', 32, ...
    'InitialLearnRate', 1e-4, ...
    'Shuffle', 'every-epoch', ...
    'ValidationFrequency', 30, ...
    'Plots', 'training-progress', ...
    'Verbose', false, ...
    'ExecutionEnvironment','auto');

net = trainNetwork(train_data, train_labels, layers, options);


% Calculate the average reward over the epochs
average_reward_per_epoch = mean(reward_history, 1);

% Plotting the convergence graph of average reward vs number of epochs
figure;
plot(1:numEpochs, average_reward_per_epoch, '-o');
xlabel('Number of Epochs');
ylabel('Average Reward');
title('Convergence of Average Reward over Epochs');
grid on;

%% Adam optimizer function with gradient clipping
function [parameters, velocity, squaredGradient] = adamOptimizer(parameters, gradients, velocity, squaredGradient, learningRate, beta1, beta2, epsilon, t, grad_clip_threshold)
    % Gradient Clipping
    gradients = min(max(gradients, -grad_clip_threshold), grad_clip_threshold);
    
    velocity = beta1 * velocity + (1 - beta1) * gradients;
    squaredGradient = beta2 * squaredGradient + (1 - beta2) * (gradients .^ 2);
    
    velocityHat = velocity / (1 - beta1^t);
    squaredGradientHat = squaredGradient / (1 - beta2^t);
    
    parameters = parameters - learningRate * velocityHat ./ (sqrt(squaredGradientHat) + epsilon);
end
%{
% Updated plotting for reward vs transmit power and block length
ptotal_range = linspace(100, 1000, 10);
blocklength_range = linspace(100, 200, 20);
average_reward_ptotal = zeros(size(ptotal_range));
average_reward_blocklength = zeros(size(blocklength_range));

for p = 1:length(ptotal_range)
    ptotal = ptotal_range(p);
    W = (randn(M, K) + 1i * randn(M, K)) * sqrt(ptotal) / sqrt(trace(W * W'));
    % Initialization of variables
    K = 4; M = 4; N = 16;
    error_prob = 10^-8;
    NF_dB = 3; N0_dB = 1; 
    total_CBL = 100; minCBL = 10;
    BW = 0.1 * 10^6; Rician_factors = 10;

    % Noise power computation
    sigma2_k = (10^(N0_dB/10)) * BW;
    user_x = [114, 132, 148, 164];
    user_y = [40, 40, 40, 40];

    % RIS parameters
    BS_loc = [0, 0];
    RIS_loc = [40, 0];
    Z0 = 50; % Impedance
    L0 = 10^(-30/10);
    pathloss_exp = 2.2;
    
    % Compute distances
    d_hk = sqrt((user_x - RIS_loc(1)).^2 + (user_y - RIS_loc(2)).^2);
    d_g = sqrt((RIS_loc(1) - BS_loc(1))^2 + (RIS_loc(2) - BS_loc(2))^2);
    % Path loss computation
    pathLoss_dB_G = L0 - 10 .* pathloss_exp .* log10(d_g);
    pathLoss_dB_h = L0 - 10 .* pathloss_exp .* log10(d_hk);
    PL_G = 10.^(pathLoss_dB_G ./ 10);
    PL_hk = 10.^(pathLoss_dB_h ./ 10);
    
    % Ricean fading channels
    h_k = zeros(N, 1, K);
    for j = 1:K
        h_k(:,:,j) = sqrt(Rician_factors/(1 + Rician_factors)) * sqrt(PL_hk(j)) * (randn(N, 1) + 1i * randn(N, 1)) + ...
                      sqrt(1/(1 + Rician_factors)) * sqrt(PL_hk(j)) * (randn(N, 1) + 1i * randn(N, 1));
    end
    
    G = sqrt(Rician_factors/(1 + Rician_factors)) * sqrt(PL_G) * (randn(N, M) + 1i * randn(N, M)) + ...
        sqrt(1/(1 + Rician_factors)) * sqrt(PL_G) * (randn(N, M) + 1i * randn(N, M));
    
    theta = rand(N, N) * 0.01;
    theta = (theta + theta') / 2;
    Theta = (1i * theta + Z0 * eye(N)) \ (1i * theta - Z0 * eye(N));
    theta_h = Theta(:);
    
    initial_channel0 = zeros(K, M);
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
    end
    
    initial_interference = zeros(K, K);
    for k_dash = 1:K
        initial_interference(:, k_dash) = initial_channel0 * W(:, k_dash);
    end
    
    % Initialize SINR, Blocklength and Adam optimizer variables
    SINR_single_connected = zeros(1, K);
    ck = max((total_CBL) .* rand(1, K), minCBL);
    t = 1;
    numEpochs = 5000;
    %reward_history = zeros(3, numEpochs);
    
    test_data = [];
    test_labels = [];
    
    % Optimization Loop
    for epoch = 1:numEpochs
        % Update SINR
        SINR_single_connected = zeros(1, K);
        for k = 1:K
            interference_power = sum(abs(initial_channel0(k,:) * W(:, setdiff(1:K, k))).^2);
            SINR_single_connected(k) = abs(initial_channel0(k,:) * W(:, k)).^2 / (sigma2_k + interference_power);
        end
        
        % Reward calculation with Rate formula
        Vk = @(gamma) 1 - (1 + gamma).^(-2);
        Ck = @(gamma) log2(1 + gamma);
        Rate_Single_Connected = ck .* Ck(SINR_single_connected) + log2(ck) - (qfuncinv(error_prob) * sqrt(ck .* Vk(SINR_single_connected)));
        reward0 = sum(Rate_Single_Connected);
            % Collect training data for Neural Network
        state1 = [abs(initial_interference(:))', angle(initial_interference(:))'];
        state2 = [ck, vecnorm(initial_channel0, 2), vecnorm(W, 2), angle(initial_channel0(:))', angle(H_tilda(:))', angle(W(:))'];
        state3 = theta(:)';
        initial_state = [state1, state2, state3, reward0]';
        
        test_data = [test_data; initial_state'];
        test_labels = [test_labels; reward0];
    
    end

    ptotal_feature = repmat(log10(ptotal), size(train_data, 1), 1);
    reward_prediction = predict(net, test_data);
    average_reward_ptotal(p) = mean(reward_prediction);
end

for q = 1:length(blocklength_range)
    ck = max((blocklength_range(q)) * rand(1, K), minCBL);

    ptotal = 1000;
    W = (randn(M, K) + 1i * randn(M, K)) * sqrt(ptotal) / sqrt(trace(W * W'));
    % Initialization of variables
    K = 4; M = 4; N = 16;
    error_prob = 10^-8;
    NF_dB = 3; N0_dB = 1; 
    total_CBL = blocklength_range(j); minCBL = 10;
    BW = 0.1 * 10^6; Rician_factors = 10;

    % Noise power computation
    sigma2_k = (10^(N0_dB/10)) * BW;
    user_x = [114, 132, 148, 164];
    user_y = [40, 40, 40, 40];

    % RIS parameters
    BS_loc = [0, 0];
    RIS_loc = [40, 0];
    Z0 = 50; % Impedance
    L0 = 10^(-30/10);
    pathloss_exp = 2.2;
    
    % Compute distances
    d_hk = sqrt((user_x - RIS_loc(1)).^2 + (user_y - RIS_loc(2)).^2);
    d_g = sqrt((RIS_loc(1) - BS_loc(1))^2 + (RIS_loc(2) - BS_loc(2))^2);
    % Path loss computation
    pathLoss_dB_G = L0 - 10 .* pathloss_exp .* log10(d_g);
    pathLoss_dB_h = L0 - 10 .* pathloss_exp .* log10(d_hk);
    PL_G = 10.^(pathLoss_dB_G ./ 10);
    PL_hk = 10.^(pathLoss_dB_h ./ 10);
    
    % Ricean fading channels
    h_k = zeros(N, 1, K);
    for j = 1:K
        h_k(:,:,j) = sqrt(Rician_factors/(1 + Rician_factors)) * sqrt(PL_hk(j)) * (randn(N, 1) + 1i * randn(N, 1)) + ...
                      sqrt(1/(1 + Rician_factors)) * sqrt(PL_hk(j)) * (randn(N, 1) + 1i * randn(N, 1));
    end
    
    G = sqrt(Rician_factors/(1 + Rician_factors)) * sqrt(PL_G) * (randn(N, M) + 1i * randn(N, M)) + ...
        sqrt(1/(1 + Rician_factors)) * sqrt(PL_G) * (randn(N, M) + 1i * randn(N, M));
    
    theta = rand(N, N) * 0.01;
    theta = (theta + theta') / 2;
    Theta = (1i * theta + Z0 * eye(N)) \ (1i * theta - Z0 * eye(N));
    theta_h = Theta(:);
    
    initial_channel0 = zeros(K, M);
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
    end
    
    initial_interference = zeros(K, K);
    for k_dash = 1:K
        initial_interference(:, k_dash) = initial_channel0 * W(:, k_dash);
    end
    
    % Initialize SINR, Blocklength and Adam optimizer variables
    SINR_single_connected = zeros(1, K);
    %ck = max((total_CBL) .* rand(1, K), minCBL);
    t = 1;
    numEpochs = 5000;
    %reward_history = zeros(3, numEpochs);
    
    test_data = [];
    test_labels = [];
    
    % Optimization Loop
    for epoch = 1:numEpochs
        % Update SINR
        SINR_single_connected = zeros(1, K);
        for k = 1:K
            interference_power = sum(abs(initial_channel0(k,:) * W(:, setdiff(1:K, k))).^2);
            SINR_single_connected(k) = abs(initial_channel0(k,:) * W(:, k)).^2 / (sigma2_k + interference_power);
        end
        
        % Reward calculation with Rate formula
        Vk = @(gamma) 1 - (1 + gamma).^(-2);
        Ck = @(gamma) log2(1 + gamma);
        Rate_Single_Connected = ck .* Ck(SINR_single_connected) + log2(ck) - (qfuncinv(error_prob) * sqrt(ck .* Vk(SINR_single_connected)));
        reward0 = sum(Rate_Single_Connected);
            % Collect training data for Neural Network
        state1 = [abs(initial_interference(:))', angle(initial_interference(:))'];
        state2 = [ck, vecnorm(initial_channel0, 2), vecnorm(W, 2), angle(initial_channel0(:))', angle(H_tilda(:))', angle(W(:))'];
        state3 = theta(:)';
        initial_state = [state1, state2, state3, reward0]';
        
        test_data = [test_data; initial_state'];
        test_labels = [test_labels; reward0];
    
    end
    blocklength_feature = repmat(log10(blocklength_range(q)), size(train_data, 1), 1);
    reward_prediction = predict(net, test_data);
    average_reward_blocklength(q) = mean(reward_prediction);
end

figure;
subplot(1,2,1);
plot(ptotal_range, average_reward_ptotal, '-o');
xlabel('Transmit Power (ptotal)');
ylabel('Average Reward');
title('Average Reward vs Transmit Power');

subplot(1,2,2);
plot(blocklength_range, average_reward_blocklength, '-o');
xlabel('Total Blocklength (CBL)');
ylabel('Average Reward');
title('Average Reward vs Blocklength');
%}
