% St-FPCA implementation
function [V_final, logs] = mbfpca(X, d, fairness, sigma, rho0, tau)
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
    % rho0: 0.1 / mmd from vanilla PCA, needed for appropriate scaling
    %
    %   
    % For missing details of ROPTLIB, check its latest documentation[1]!
    % [1] 
    

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
    
    % Center the data matrix such that the sum of x_i's (row-vectors) is zero
    X = X - mean(X);
%     X = center(X);
    [~, p] = size(X);
    
    % Define total covariance matrix
    A = cov(X);
    tot_expvar = trace(A);
    
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
    
    % main objective (variance explained)
    exp_var = @(V) trace(A * (V.main * V.main')) / tot_expvar;
    exp_var_egrad = @(V) 2 * A * V.main / tot_expvar;
%     exp_var = @(V) 0;
%     exp_var_egrad = @(V) zeros(p, d);
    
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
    SolverParams.IsCheckGradHess = 0;
    SolverParams.Stop_Criterion = 1;    % Norm of the gradient
    SolverParams.Max_Iteration = 300;
    
%     eps = 1e-3;
    SolverParams.Tolerance = 1e-1;
    eps_min = 1e-6;
%     SolverParams.Tolerance = eps_min;
    theta_eps = (eps_min / SolverParams.Tolerance)^(1/5);
    
    % Other Parameters
    rho = rho0;
    rho_max = 1e10;
    theta_rho = 2;
        
    dist_min = 1e-6;
    
    maxiter = 100;
        
    
    %% Initialization
%     tmp = pca(X);
%     V_prev.main = tmp(:, 1:d);
    V_prev = mbfpca_auxiliary([X Y Z], d, fairness, sigma);

    %% Initialize empty lists for plotting
    exp_vars = [];
    rhos = [];
    mmds = [];
    mmds_ = [];
    distances = [];
    
%     mmd_constraint(V_prev)
    
%     V_pca = pca(X);
%     V_pca = V_pca(:, 1:d);
%     V_prev.main = V_pca;
    
    exp_vars = [exp_vars, exp_var(V_prev)];
    mmds = [mmds, mmd_constraint(V_prev)];
    mmds_ = [mmds_, mmd_constraint_(V_prev)];
    rhos = [rhos, rho];
    distances = [distances, 0];

    tStart = cputime;
    tStart_ = tic;
    iter = 1;
    logs.fail = false;
    restart = true;
    %% main loop!
    while iter <= maxiter + 1
        % Define iteration-wise cost
        fhandle = @(V) main_cost(V, rho, exp_var, mmd_constraint, mmd_constraint_);
        f_egrad = @(V) main_cost_egrad(V, rho, exp_var_egrad, mmd_constraint_egrad, mmd_constraint_egrad_);
        
        % Solve subproblem
        % What does "HasHHR = 0" mean?
        V = DriverOPT(fhandle, f_egrad, [], [], SolverParams, ManiParams, 0, V_prev);
%         V = DriverOPT(fhandle, [], [], [], SolverParams, ManiParams, 0, V_prev);
        
        dist = norm(V.main - V_prev.main, 'fro');
        mmd_violation = (mmd_constraint(V) > tau) || (mmd_constraint_(V) > tau);
        
        % Update logs
        exp_vars = [exp_vars, exp_var(V)];
        mmds = [mmds, mmd_constraint(V)];
        mmds_ = [mmds_, mmd_constraint_(V)];
        rhos = [rhos, rho];
        distances = [distances, dist];
        
        % (Main) termination criterion
        % Terminate if the iterates are close enough,
        % AND the fairness constraints are both satisfied!
        if (dist <= dist_min) && (SolverParams.Tolerance <= eps_min)
            if ~mmd_violation
                sprintf("At iteration %d, St-FPCA terminated properly!", iter)
                if iter == 1
                    logs.walltime = toc(tStart_)
                    logs.cputime = cputime - tStart;
                    logs.exp_vars = exp_vars;
                    logs.mmds = mmds;
                    logs.mmds_ = mmds_;
                    logs.rhos = rhos;
                    logs.distances = distances;
                end
                break;
            else
                sprintf("At iteration %d, St-FPCA could've terminated improperly!", iter)
                sprintf("%d  %d  %d", exp_vars(end), mmds(end), tau)
            end
        end
        
        % (Heuristic) termination criterion...
        % Restart the loop (with relaxed threshold) if rho is too large...
        tmp = rho > rho_max || iter == maxiter + 1;
        
        if tmp
            sprintf("At iteration %d, St-FPCA terminated improperly!", iter)
            sprintf("%d  %d  %d", exp_vars(end), mmds(end), tau)
            break;
            
            % algorithm restart?
            if restart
                sprintf("Restarting loop with relaxed tau: %d -> %d", tau, tau * 5)
                tau = tau * 5;
                % Reset other parameters
                iter = 1;
                SolverParams.Tolerance = 1e-6;
                rho = rho0;
                
                disp("Searching for new init...")
                V_prev.main = orth(randn(p, d));
            %     mmd_constraint(V_prev)
                for j = 1:10
                    W.main = orth(randn(p, d));
                    if mmd_constraint(W) < mmd_constraint(V_prev)
                        V_prev = W;
                    end
                end
%                 V_prev = mbf_auxiliary([X Y Z], d, fairness, sigma);
                disp("Done! Now the main loop:")

                logs.fail = true;
                restart = false;

                continue;
            else
                break;
            end
            
        end
        
        
        % Update tolerance
        SolverParams.Tolerance = max(eps_min, theta_eps * SolverParams.Tolerance);
        
        % Update penalty coefficients
        if mmd_violation
            rho = rho * theta_rho;
        end
        
        V_prev.main = V.main;
        
        iter = iter + 1;
    end
    
    %% Save stuff
    logs.walltime = toc(tStart_);
    logs.cputime = cputime - tStart;
    logs.exp_vars = exp_vars;
    logs.mmds = mmds;
    logs.mmds_ = mmds_;
    logs.rhos = rhos;
    logs.distances = distances;
    
    V_final = V;
end

function [output, V] = main_cost(V, rho, exp_var, mmd_constraint, mmd_constraint_)
    output = -exp_var(V) + rho * (mmd_constraint(V) + mmd_constraint_(V));
end


function [output, V] = main_cost_egrad(V, rho, exp_var_egrad, mmd_constraint_egrad, mmd_constraint_egrad_)
    output.main = -exp_var_egrad(V) + rho * (mmd_constraint_egrad(V) + mmd_constraint_egrad_(V));
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