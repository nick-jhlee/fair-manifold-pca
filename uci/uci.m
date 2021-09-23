names = {'COMPAS', 'German', 'Adult'};

fairness = 'DP';
% d = 2;
d = 10;

% tau = 1e-3;
tau = 1e-6;

for name_num = 1:3
    clc;
    for split = 1:10
        %% Load datas
        X_train = table2array(readtable(sprintf('../datasets/%s/train_%d.csv', names{name_num}, split-1)));
        Y_train = X_train(:, end-1);
        Z_train = X_train(:, end);
        A_train = cov(X_train(:, 1:end-2));

        X = table2array(readtable(sprintf('../datasets/%s/test_%d.csv', names{name_num}, split-1)));
        Y = X(:, end-1);
        Z = X(:, end);
        n1 = sum(Z);
        n2 = sum(Z == 0);
        X = X(:, 1:end-2);
        A = cov(X);

        %% Obtain PCA and sigma
        V_pca = pca(X_train(:, 1:end-2));
        V_pca = V_pca(:, 1:d);

        % Obtain sigma
        sigma = sqrt(median(pdist(X_train(:, 1:end-2)*V_pca, 'squaredeuclidean'))/2);
        m_ = mmd(X_train(Z_train == 1, 1:end-2)*V_pca, X_train(Z_train == 0, 1:end-2)*V_pca, sigma);

        %% Initialize rho
        rho0 = 1/m_;

        %% Run St-FPCA
        [V, logs] = mbfpca(X_train, d, fairness, sigma, rho0, tau);
%             V = mbfpca_auxiliary(X_train, d, fairness, sigma);
        V = V.main;

        % Save_loading matrix
        writematrix(V, sprintf('%s/MBFPCA_V_%d.csv', names{name_num}, split-1))
    end
end
