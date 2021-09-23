names = {'COMPAS', 'German', 'Adult'};

mmds_pca_train = zeros(10, 3);
exp_vars_pca_train = zeros(10, 3);
mmds_pca_test = zeros(10, 3);
exp_vars_pca_test = zeros(10, 3);

accs_pca = zeros(10, 3);
DPs_pca = zeros(10, 3);
EOPs_pca = zeros(10, 3);
EODs_pca = zeros(10, 3);

% d = 10;
d = 2;

%% dimensions considered
for name_num = 1:3
    for split = 1:10
        %% Load datas
        X_train = table2array(readtable(sprintf('../datasets/%s/train_%d.csv', names{name_num}, split-1)));
        Y_train = X_train(:, end-1);
        Z_train = X_train(:, end);
        X_train = X_train(:, 1:end-2);
        A_train = cov(X_train);
        
        X = table2array(readtable(sprintf('../datasets/%s/test_%d.csv', names{name_num}, split-1)));
        Y = X(:, end-1);
        Z = X(:, end);
        n1 = sum(Z);
        n2 = sum(Z == 0);
        X = X(:, 1:end-2);
        A = cov(X);
        
        %% Obtain PCA and sigma
        V_pca = pca(X_train);
        V_pca = V_pca(:, 1:d);
        
        % Obtain sigma
        sigma = sqrt(median(pdist(X_train*V_pca, 'squaredeuclidean'))/2);
        
        
        %% Store FPCA results
        % train
        mmds_pca_train(split, name_num) = mmd(X_train(Z_train==1,:)*V_pca, X_train(Z_train==0,:)*V_pca, sigma);
        exp_vars_pca_train(split, name_num) = 100 * trace(V_pca'*A_train*V_pca)/trace(A_train);
        % test
        mmds_pca_test(split, name_num) = mmd(X(Z==1,:)*V_pca, X(Z==0,:)*V_pca, sigma);
        exp_vars_pca_test(split, name_num) = 100 * trace(V_pca'*A*V_pca)/trace(A);
        
        % fairness metrics
        [acc, DP, EOP, EOD] = fairness_metric(X*V_pca, Y, Z);
        accs_pca(split, name_num) = acc;
        DPs_pca(split, name_num) = DP;
        EOPs_pca(split, name_num) = EOP;
        EODs_pca(split, name_num) = EOD;
    end
end

writematrix(exp_vars_pca_train, 'pca/exp_vars_train.csv')
writematrix(mmds_pca_train, 'pca/mmds_train.csv')
writematrix(exp_vars_pca_test, 'pca/exp_vars_test.csv')
writematrix(mmds_pca_test, 'pca/mmds_test.csv')

writematrix(accs_pca, 'pca/accs.csv')
writematrix(DPs_pca, 'pca/DPs.csv')
writematrix(EOPs_pca, 'pca/EOPs.csv')
writematrix(EODs_pca, 'pca/EODs.csv')

