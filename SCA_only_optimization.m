clc;
clear;

% Parameters
sigma2 = 1;        % Noise power
K = 4;
M = 4;
N = 16;
error_prob = 10^-6;
ptotal = 10^-2;
N0_dB = -174;       %(dBm/Hz)
total_CBL = 200;
minCBL = 20;
BW = 2*10^6;
Rician_factors = 1;
BS_height = 12.5;
Z0 = 50;
alpha_tcheb = 0.8;


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

%% Ricean fading channels
h_k = zeros(N, 1, K);
for j = 1:K
    h_k(:,:,j) = sqrt(Rician_factors/(1+Rician_factors)) * sqrt(PL_hk(j)) * (randn(N, 1) + 1i * randn(N, 1)) + sqrt(1/(1+Rician_factors)) * sqrt(PL_hk(j)) * (randn(N, 1) + 1i * randn(N, 1));
end

G = sqrt(Rician_factors/(1+Rician_factors)) * sqrt(PL_G) * (randn(N, M) + 1i * randn(N, M)) + sqrt(1/(1+Rician_factors)) * sqrt(PL_G) * (randn(N, M) + 1i * randn(N, M));

%% Beamforming and channel initialization

%W = (randn(M, K) + 1i * randn(M, K)) * 0.01; % Smaller initialization for W
s = randn(K, 1) + 1i * randn(K, 1);
s = s / norm(s);
%W = W / sqrt(trace(W * W') / ptotal);

p_vec = (ptotal/K).*ones(1,K);    %equal power allocation

theta = rand(N, N) ; % Smaller initialization for theta
theta = (theta + theta') / 2;
Theta = (1i * theta + Z0 * eye(N)) \ (1i * theta - Z0 * eye(N));
theta_h = Theta(:);

initial_channel0 = zeros(K, M);
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

interference_power = zeros(1, K);
for k = 1:K
    interference_power(k) = sum(p_vec(setdiff(1:K, k)).*(abs(initial_channel0(k,:) * W_init(:, setdiff(1:K, k))).^2));
end


SINR_fully_connected = zeros(1, K);
for k = 1:K
    interference_p = sum(p_vec(setdiff(1:K, k)).*(abs(initial_channel0(k,:) * W_init(:, setdiff(1:K, k))).^2));
    SINR_fully_connected(k) = (p_vec(k)*(abs(initial_channel0(k,:) * W_init(:, k)).^2)) / (sigma2_k + interference_p);
end

%% Reward calculation
Vk = @(gamma) 1 - (1 + gamma).^(-2);
Ck = @(gamma) log2(1 + gamma);
mk = max((total_CBL)/K, minCBL) * ones(1,K);
Rate_Fully_Connected = mk .* Ck(SINR_fully_connected) + log2(mk) - (qfuncinv(error_prob) * sqrt(mk .* Vk(SINR_fully_connected)));
sumrate0 = sum(Rate_Fully_Connected);
% Q inverse

% Add after your initializations
max_iter = 20;         % Maximum number of SCA iterations
tolerance = 1e-4;      % Tolerance for convergence
alpha_tcheb = 0.8;     % Tchebyshev weight

L_total_star = sumrate0;   % Using initial sumrate as utopia point
m_total_star = minCBL*K;    % Minimum CBL (sum of minCBL for all users)
mk_fixed = mk;
p_opt = p_vec';    % Initialize optimized p
W_fixed = W_init; % Fix beamforming for now
clip_value = 1e3;
for iter = 1:max_iter
    % Compute SINR with current p_opt
    SINR_current = zeros(1, K);
    for k = 1:K
        interf = sum(p_opt(setdiff(1:K,k))'.*(abs(initial_channel0(k,:)*W_fixed(:,setdiff(1:K,k))).^2));
        signal = p_opt(k)*(abs(initial_channel0(k,:)*W_fixed(:,k)).^2);
        SINR_current(k) = signal / (sigma2_k + interf);
    end
    %mk_fixed = mk;
    m_total = sum(mk);
    % Compute current L_total and gradients

    L_plus = sum(mk .* log2(arrayfun(@(k) sigma2_k + sum(p_opt(setdiff(1:K,k))'.*(abs(initial_channel0(k,:)*W_fixed(:,setdiff(1:K,k))).^2)) + ...
        p_opt(k)*(abs(initial_channel0(k,:)*W_fixed(:,k)).^2), 1:K)) + log2(mk));
    
    L_minus = sum(mk .* log2(arrayfun(@(k) sigma2_k + sum(p_opt(setdiff(1:K,k))'.*(abs(initial_channel0(k,:)*W_fixed(:,setdiff(1:K,k))).^2)), 1:K)) + ...
        (sqrt(mk) / log(2)) .* qfuncinv(error_prob));
    
    L_dash = sum(qfuncinv(error_prob) * (sqrt(mk.*Vk(SINR_current))./2) .*(1+(mk_fixed./mk)));

    L_total_now = L_plus - L_minus;

    L_minus_at_popt = 0;
    grad_L_minus = zeros(K, 1);
    for k = 1:K
        interf_k = 0;
        for j = 1:K
            if j ~= k
                interf_k = interf_k + p_opt(j) * (abs(initial_channel0(k,:) * W_fixed(:,j)))^2;
            end
        end
        I_k = interf_k + sigma2_k;
        if I_k < 1e-13
            I_k = 1e-13;  % Prevent division by zero
        end
        L_minus_at_popt = L_minus_at_popt + mk(k)*log(I_k)/log(2) + (sqrt(mk(k))/log(2))*qfuncinv(error_prob);
        
        % Gradient w.r.t. each p_j
        for j = 1:K
            if j ~= k
                dI_dpj = (abs(initial_channel0(k,:) * W_fixed(:,j)))^2;
                grad = mk(k) * (1 / (log(2) * I_k)) * dI_dpj;
    
                % Clip gradient
                grad = min(max(grad, -clip_value), clip_value);
    
                % Accumulate
                grad_L_minus(j) = grad_L_minus(j) + grad;
            end
        end
    end


    % Successive convex approximation (SCA) optimization
    cvx_begin 
        variable p(K) nonnegative
        variable mu
        expressions interf_power(K) L_plus_app(K)
        
        for k = 1:K
            interference_k = 0;
            for j = 1:K
                if j ~= k
                    interference_k = interference_k + p(j) * (abs(initial_channel0(k,:) * W_init(:,j)))^2;
                end
            end
            interf_power(k) = interference_k + sigma2_k;
        end
        
        % Approximated L_plus and L_minus
        for k = 1:K
            L_plus_app(k) = mk(k) .* ((log(interf_power(k) + p(k)*(abs(initial_channel0(k,:)*W_fixed(:,k)).^2)))/log(2)) + log2(mk(k));
        end 

        L_plus_approx = sum(L_plus_app);

        %L_minus_fixed = sum(mk(:) .* (log(interf_power)/log(2)) + (sqrt(mk(:))/log(2).*qfuncinv(error_prob))); % fixed in SCA

        L_total_approx = L_plus_approx - dot(grad_L_minus, p) - L_minus_at_popt + dot(grad_L_minus, p_opt);
        
        minimize(mu)
        
        subject to
            sum(p) <= ptotal;     % total power constraint
            p>=1e-6;
            alpha_tcheb/L_total_star * (L_total_star - L_total_approx) <= mu;
    cvx_end
    
    % Check convergence
    if norm(p - p_opt) < tolerance
        disp(['Converged in ', num2str(iter), ' iterations.']);
        break;
    end

    % Update
    p_opt = p;

      cvx_begin 
        variable mk_new(1, K) nonnegative
        variable mu
        expressions L_minus_dash(K) L_plus_app(k)
        
        for k = 1:K
            interference_k = 0;
            for j = 1:K
                if j ~= k
                    interference_k = interference_k + p(j) * (abs(initial_channel0(k,:) * W_init(:,j)))^2;
                end
            end
            interf_power(k) = interference_k + sigma2_k;
        end
        
        % Approximated L_plus and L_minus
        for k = 1:K
            L_plus_app(k) = mk_new(k) .* ((log(interf_power(k) + p_opt(k)*(abs(initial_channel0(k,:)*W_fixed(:,k)).^2)))/log(2)) + log(mk_new(k))./log(2);
        end 

        L_plus_approx = sum(L_plus_app);
        
        L_minus_dash = sum(qfuncinv(error_prob).*(sqrt(Vk(SINR_current).*mk_fixed)./2).*(1+(mk_new./mk_fixed)));
        %L_minus_fixed = sum(mk(:) .* (log(interf_power)/log(2)) + (sqrt(mk(:))/log(2).*qfuncinv(error_prob))); % fixed in SCA

        L_total_approx = L_minus_dash - L_plus_approx;
        
        minimize(mu)
        
        subject to
            sum(mk_new) <= total_CBL;     % total CBL constraint
            mk_new >= minCBL;
            (1-alpha_tcheb)/(m_total_star*(m_total-m_total_star)) <= mu;
            L_total_approx <= L_total_star*(mu/alpha_tcheb - 1);
    cvx_end

    % Check convergence
    if norm(mk_new - mk) < tolerance
        disp(['Converged in ', num2str(iter), ' iterations.']);
        break;
    end

    mk = mk_new;
end

disp('Optimized power allocation p_opt:');
disp(p_opt);
disp('optimized CBL:');
disp(mk);
