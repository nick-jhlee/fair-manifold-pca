mmds_pca_train = zeros(10, 18);
exp_vars_pca_train = zeros(10, 18);
mmds_pca_test = zeros(10, 18);
exp_vars_pca_test = zeros(10, 18);

fairness = 'DP';
d = 5;


%% dimensions considered
folder_num = 6;
for tau = [1 1e-1 1e-2 1e-3 1e-4 1e-5 1e-6 1e-7]
    tmp = 1;
    
    mmds_mbfpca_train = zeros(10, 18);
    exp_vars_mbfpca_train = zeros(10, 18);
    mmds_mbfpca_test = zeros(10, 18);
    exp_vars_mbfpca_test = zeros(10, 18);
    runtimes_mbfpca = zeros(10, 18);
    for p = 20:10:100
        clc;
        for split = 1:10
            sprintf('%f %d %d', tau, p, split)
            %% Load datas
            X_train = table2array(readtable(sprintf('%d/train_%d.csv', p, split)));
            Z_train = X_train(:, end);
            A_train = cov(X_train(:, 1:end-2));

            X = table2array(readtable(sprintf('%d/test_%d.csv', p, split)));
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

            % Store PCA results
            if tau == 1
                % train
                mmds_pca_train(split, tmp) = mmd(X_train(Z == 1, 1:end-2)*V_pca, X_train(Z == 0, 1:end-2)*V_pca, sigma);
                exp_vars_pca_train(split, tmp) = 100 * trace(V_pca'*A_train*V_pca)/trace(A_train);
                % test
                mmds_pca_test(split, tmp) = mmd(X(Z == 1, :)*V_pca, X(Z == 0, :)*V_pca, sigma);
                exp_vars_pca_test(split, tmp) = 100 * trace(V_pca'*A*V_pca)/trace(A);
            end

            % Initialize rho
            rho0 = 1/m_;

            %% Run MbF-PCA
            [V, logs] = mbfpca(X_train, d, fairness, sigma, rho0, tau);
    %         [V, logs] = mbfpca_auxiliary(X_train, d, fairness, sigma, rho0);
            V = V.main;

            % Store St-FPCA results
            % train
            mmds_mbfpca_train(split, tmp) = mmd(X_train(Z == 1, 1:end-2)*V, X_train(Z == 0, 1:end-2)*V, sigma);
            exp_vars_mbfpca_train(split, tmp) = 100 * trace(V'*A_train*V)/trace(A_train);
            % test
            mmds_mbfpca_test(split, tmp) = mmd(X(Z == 1,:)*V, X(Z == 0,:)*V, sigma);
            exp_vars_mbfpca_test(split, tmp) = 100 * trace(V'*A*V)/trace(A);
            % runtime
            runtimes_mbfpca(split, tmp) = logs.cputime;
        end

        tmp = tmp + 1;

        % Save in each loop
        writematrix(mmds_mbfpca_train, sprintf('mbfpca_%d/mmds_train.csv', folder_num))
        writematrix(exp_vars_mbfpca_train, sprintf('mbfpca_%d/exp_vars_train.csv', folder_num))
        writematrix(mmds_mbfpca_test, sprintf('mbfpca_%d/mmds_test.csv', folder_num))
        writematrix(exp_vars_mbfpca_test, sprintf('mbfpca_%d/exp_vars_test.csv', folder_num))
        writematrix(runtimes_mbfpca, sprintf('mbfpca_%d/runtimes.csv', folder_num))

        if tau == 1
            writematrix(exp_vars_pca_train, 'pca/exp_vars_train.csv')
            writematrix(mmds_pca_train, 'pca/mmds_train.csv')
            writematrix(exp_vars_pca_test, 'pca/exp_vars_test.csv')
            writematrix(mmds_pca_test, 'pca/mmds_test.csv')
        end
    end
    
    folder_num = folder_num + 1;
end

