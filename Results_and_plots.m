% Set random seed
seed = randi([1, 200]);
rng(seed);

% Initialization of variables
K = 4; M = 4; N = 16;
error_prob = 10^-8; ptotal = 10^3;
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
hk = zeros(N,K);
for j = 1:K
    h_k(:,:,j) = sqrt(Rician_factors/(1 + Rician_factors)) * sqrt(PL_hk(j)) * (randn(N, 1) + 1i * randn(N, 1)) + ...
                  sqrt(1/(1 + Rician_factors)) * sqrt(PL_hk(j)) * (randn(N, 1) + 1i * randn(N, 1));
end

G = sqrt(Rician_factors/(1 + Rician_factors)) * sqrt(PL_G) * (randn(N, M) + 1i * randn(N, M)) + ...
    sqrt(1/(1 + Rician_factors)) * sqrt(PL_G) * (randn(N, M) + 1i * randn(N, M));

% Initialize W, Theta and CBL
W = (randn(M, K) + 1i * randn(M, K)) * 0.01;
W = W / sqrt(trace(W * W') / ptotal); % Normalize transmit power

theta = rand(N,1) * 0.01;
Theta = exp(1i*theta);
theta_h = Theta(:);

initial_channel0 = zeros(K, M);
H_tilda = zeros(N, M, K);

for k = 1:K
    hk(:,k) = h_k(:,:,k);
    ak = diag(hk(:,k)')*G;
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
epsilon = 1e-8; numEpochs = 100;
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
    'OutputFcn', @(info)saveTrainingLoss1(info), ...
    'ExecutionEnvironment','auto');
net = trainNetwork(train_data, train_labels, layers, options);
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
epsilon = 1e-8; numEpochs = 100;
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
    'ExecutionEnvironment','auto', ...
    'OutputFcn', @(info)saveTrainingLoss2(info));

net = trainNetwork(train_data, train_labels, layers, options);


% Initialization of variables
K = 4; M = 4; N = 16;
error_prob = 10^-8; ptotal = 10^3;
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
d_direct = sqrt((user_x - BS_loc(1)).^2 + (user_y - BS_loc(2)).^2);
% Path loss computation
pathLoss_dB_G = L0 - 10 .* pathloss_exp .* log10(d_g);
pathLoss_dB_h = L0 - 10 .* pathloss_exp .* log10(d_hk);
pathLoss_dB_g = L0 - 10 .* pathloss_exp .* log10(d_direct);
PL_G = 10.^(pathLoss_dB_G ./ 10);
PL_hk = 10.^(pathLoss_dB_h ./ 10);
PL_gk = 10.^(pathLoss_dB_g ./ 10); 
% Ricean fading channels
h_k = zeros(N, 1, K);
hk = zeros(N,K);
for j = 1:K
    h_k(:,:,j) = sqrt(Rician_factors/(1 + Rician_factors)) * sqrt(PL_hk(j)) * (randn(N, 1) + 1i * randn(N, 1)) + ...
                  sqrt(1/(1 + Rician_factors)) * sqrt(PL_hk(j)) * (randn(N, 1) + 1i * randn(N, 1));
end
g_k = zeros(M, 1, K);
for j = 1:K
    g_k(:,:,j) = sqrt(Rician_factors/(1 + Rician_factors)) * sqrt(PL_gk(j)) * (randn(M, 1) + 1i * randn(M, 1)) + ...
                  sqrt(1/(1 + Rician_factors)) * sqrt(PL_gk(j)) * (randn(M, 1) + 1i * randn(M, 1));
end
G = sqrt(Rician_factors/(1 + Rician_factors)) * sqrt(PL_G) * (randn(N, M) + 1i * randn(N, M)) + ...
    sqrt(1/(1 + Rician_factors)) * sqrt(PL_G) * (randn(N, M) + 1i * randn(N, M));

% Initialize W, Theta and CBL
W = (randn(M, K) + 1i * randn(M, K)) * 0.01;
W = W / sqrt(trace(W * W') / ptotal); % Normalize transmit power


initial_channel0 = zeros(K, M);
H_tilda = zeros(M, K);

for k = 1:K
    initial_channel0(k,:) = g_k(:,:,k);
    H_tilda(:,k) = g_k(:,:,k);
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
epsilon = 1e-8; numEpochs = 100;
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
    'OutputFcn', @(info)saveTrainingLoss3(info), ...
    'ExecutionEnvironment','auto');

net = trainNetwork(train_data, train_labels, layers, options);
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
epsilon = 1e-8; numEpochs = 100;
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
        SINR_single_connected(k) = abs(initial_channel0(k,:) * W(:, k)).^2 / (sigma2_k);
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
    'ExecutionEnvironment','auto', ...
    'OutputFcn', @(info)saveTrainingLoss4(info));

net = trainNetwork(train_data, train_labels, layers, options);

% Load saved losses for each parameter set
load('loss1.mat', 'loss1');
load('loss2.mat', 'loss2');
load('loss3.mat', 'loss3');
load('loss4.mat', 'loss4');
% Smooth the loss data using a moving average
smooth_loss1 = smoothdata(loss1, 'movmean', 10); % Smooth over a window of 10 epochs
smooth_loss2 = smoothdata(loss2, 'movmean', 10);
smooth_loss3 = smoothdata(loss3, 'movmean', 10); % Smooth over a window of 10 epochs
smooth_loss4 = smoothdata(loss4, 'movmean', 10);
% Plot both smoothed losses on the same figure
figure;
plot(smooth_loss1, 'b-', 'LineWidth', 1.5);
hold on;
plot(smooth_loss2, 'r--', 'LineWidth', 1.5);
plot(smooth_loss3, 'm-.', 'LineWidth', 1.5);
plot(smooth_loss4, 'g-', 'LineWidth', 1.5);
xlabel('Epochs (Iterations)');
ylabel('Loss (RMSE)');
legend('Fully RIS NOMA', 'Single RIS NOMA','No RIS', 'Fully RIS OMA', 'Location','northeast');
title('Comparison of Smoothed Loss/RMSE vs Epochs');
grid on;
hold off;

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

% Function to capture training progress (for first parameter set)
function stop = saveTrainingLoss1(info)
    persistent loss1 
    stop = false;
    if info.State == "iteration"
        % Capture and store loss at each iteration for first training
        loss1(end + 1) = info.TrainingLoss;
        
    elseif info.State == "done"
        save('loss1.mat', 'loss1'); % Save the loss history to a file
    end
end

% Function to capture training progress (for second parameter set)
function stop = saveTrainingLoss2(info)
    persistent loss2
    stop = false;
    if info.State == "iteration"
        % Capture and store loss at each iteration for second training
        loss2(end + 1) = info.TrainingLoss;
    elseif info.State == "done"
        save('loss2.mat', 'loss2'); % Save the loss history to a file
    end
end

function stop = saveTrainingLoss3(info)
    persistent loss3 
    stop = false;
    if info.State == "iteration"
        % Capture and store loss at each iteration for first training
        loss3(end + 1) = info.TrainingLoss;
        
    elseif info.State == "done"
        save('loss3.mat', 'loss3'); % Save the loss history to a file
    end
end

function stop = saveTrainingLoss4(info)
    persistent loss4 
    stop = false;
    if info.State == "iteration"
        % Capture and store loss at each iteration for first training
        loss4(end + 1) = info.TrainingLoss;
        
    elseif info.State == "done"
        save('loss4.mat', 'loss4'); % Save the loss history to a file
    end
end