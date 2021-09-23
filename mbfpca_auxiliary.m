% MbF-PCA *without* main objective (i.e. Eq. 10 of our preprint)
function V_init = mbfpca_auxiliary(X, d, fairness, sigma)
    % X: (n x (p + 2)) data matrix, each row is a p-dimensional data sample
    % X(:, -2) is the DOWNSTREAM TASK LABEL (binary!)
    % X(:, -1) is the SENSITIVE GROUP LABEL (binary!)
    % (binary as in {0, 1})
    %
    % d: desired output dimension
    %
    % fairness: str, type of fairness to pursue (DEM, EOP, EOD)
    %
    % sigma: bandwidth for rbf MMD (use median heuristic!)
    %
    %   
    % For missing details of ROPTLIB, check its latest documentation[1]!
    
    %% Data Preprocessing
    % Extract Y and Z
    Y = X(:, end-1);
    Z = X(:, end);
    if ~prod((Y == 1) | (Y == 0))
        error('Some entry(s) of Y is not 0 nor 1')
    end
    if ~prod((Z == 1) | (Z == 0))
        error('Some entry(s) of Z is not 0 nor 1')
    end
    X = X(:, 1:end-2);
    
    % Center the data matrix such that the sum of x_i's is zero
    X = center(X);
    [~, p] = size(X);
    
    % Define total covariance matrix
    A = cov(X);
    
    % Partition X appropriately
    if strcmp(fairness, 'EOD')
        X1 = X((Y == 1) & (Z == 1), :);
        X2 = X((Y == 1) & (Z == 0), :);
        X1_ = X((Y == 0) & (Z == 1), :);
        X2_ = X((Y == 0) & (Z == 0), :);
    else
        X1_ = nan;
        X2_ = nan;
        if strcmp(fairness, 'EOP')
            X1 = X((Y == 1) & (Z == 1), :);
            X2 = X((Y == 1) & (Z == 0), :);
        elseif strcmp(fairness, 'DEM') || strcmp(fairness, 'DP')
            X1 = X(Z == 1, :);
            X2 = X(Z == 0, :);
        else
            error('%s fairness not yet implemented', fairness)
        end
    end
    
    %% Define functions
    % Define mmd fairness constraints
    mmd_constraint = @(V) mmd_penalty(V.main, X1, X2, sigma);
    mmd_constraint_egrad = @(V) mmd_penalty_egrad(V.main, X1, X2, sigma);
    
    if strcmp(fairness, 'EOD')
        mmd_constraint_ = @(V) mmd_penalty(V.main, X1_, X2_, sigma);
        mmd_constraint_egrad_ = @(V) mmd_penalty_egrad(V.main, X1_, X2_, sigma);
    else
        mmd_constraint_ = @(V) 0;
        mmd_constraint_egrad_ = @(V) zeros(p, d);
    end
    
    %% Hyperparameters
    % Manifold Parameters (Table 39 of [1])
    ManiParams.IsCheckParams = 1;
    ManiParams.name = 'Stiefel';
    ManiParams.n = p;
    ManiParams.p = d;
    ManiParams.ParamSet = 1;
    
    % Solver Parameters (Table 2 of [1])
%     SolverParams.method = 'RTRNewton';
%     SolverParams.method = 'RSD';
    SolverParams.method = 'RBFGS';
    SolverParams.IsCheckParams = 1;
    SolverParams.Verbose = 0;
    
    % Random initialization
    disp("Searching for auxiliary auxiliary init...")
    V_prev.main = orth(randn(p, d));
%     mmd_constraint(V_prev)
    for j = 1:10
        W.main = orth(randn(p, d));
        if trace(A * (V_prev.main * V_prev.main')) < trace(A * (W.main * W.main'))
%         if mmd_constraint(W) < mmd_constraint(V_prev)
            V_prev = W;
        end
    end
    disp("Done")
    
    fhandle = @(V) main_cost(V, mmd_constraint, mmd_constraint_);
    f_egrad = @(V) main_cost_egrad(V, mmd_constraint_egrad, mmd_constraint_egrad_);
        
    V_init = DriverOPT(fhandle, f_egrad, [], [], SolverParams, ManiParams, 0, V_prev);
end

function [output, V] = main_cost(V, mmd_constraint, mmd_constraint_)
    output = mmd_constraint(V) + mmd_constraint_(V);
end


function [output, V] = main_cost_egrad(V, mmd_constraint_egrad, mmd_constraint_egrad_)
    output.main = mmd_constraint_egrad(V) + mmd_constraint_egrad_(V);
end

function y = mmd_penalty(V, X1, X2, sigma)
    y = mmd(X1*V, X2*V, sigma);
end

function y = mmd_penalty_egrad(V, X1, X2, sigma)
    m = size(X1, 1);
    n = size(X2, 1);
    
    X1_ = X1*V;
    X2_ = X2*V;
    
    % Kernel gram matrices
    K1 = rbf(X1_, X1_, sigma);
    K2 = rbf(X2_, X2_, sigma);
    K12 = rbf(X1_, X2_, sigma);

    y1 = (1/m^2) * (X1' * (diag(ones(1,m)*K1) - K1) * X1);
    y2 = (1/n^2) * (X2' * (diag(ones(1,n)*K2) - K2) * X2);
    y3 = (1/(m*n)) * ((X1' * diag(K12*ones(n,1)) * X1) + (X2' * diag(ones(1, m)*K12) * X2) - (X1' * K12 * X2) - (X2' * K12' * X1));
    
    y = -(2 / sigma^2) * (y1 + y2 - y3) * V;
end

function K = rbf(X1, X2, sigma)
    K = pdist2(X1, X2, 'squaredeuclidean');
    K = exp(-(1/(2*sigma^2)) * K);
%     if clear_diag
%         K = K - diag(diag(K));
%     end
end