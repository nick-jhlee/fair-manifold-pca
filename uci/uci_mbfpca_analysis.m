names = {'COMPAS', 'German', 'Adult'};

mmds_mbfpca_train = zeros(10, 3);
exp_vars_mbfpca_train = zeros(10, 3);
mmds_mbfpca_test = zeros(10, 3);
exp_vars_mbfpca_test = zeros(10, 3);

accs_mbfpca = zeros(10, 3);
DPs_mbfpca = zeros(10, 3);
EOPs_mbfpca = zeros(10, 3);
EODs_mbfpca = zeros(10, 3);

% d = 2;
d = 10;

%% dimensions considered
for name_num = 3:3
% for name_num = 1:3
    for split = 1:10
        %% Load datas
        X_train = table2array(readtable(sprintf('../datasets/%s/train_%d.csv', names{name_num}, split-1)));
        Y_train = X_train(:, end-1);
        Z_train = X_train(:, end);
        X_train = X_train(:, 1:end-2);
        A_train = cov(X_train);
        
        X = table2array(readtable(sprintf('../datasets/%s/test_%d.csv', names{name_num}, split-1)));
%         V_mbfpca = table2array(readtable(sprintf('%s/mbfpca_V_%d.csv', names{name_num}, split-1)));
        Y = X(:, end-1);
        Z = X(:, end);
        n1 = sum(Z);
        n2 = sum(Z == 0);
        X = X(:, 1:end-2);
        A = cov(X);
        V_mbfpca = table2array(readtable(sprintf('%s/mbfpca_V_%d.csv', names{name_num}, split-1)));
        
        %% Obtain PCA and sigma
        V_pca = pca(X_train);
        V_pca = V_pca(:, 1:d);
        
        % Obtain sigma
        sigma = sqrt(median(pdist(X_train*V_pca, 'squaredeuclidean'))/2);
        
        
        %% Store FPCA results
        % train
        mmds_mbfpca_train(split, name_num) = mmd(X_train(Z_train==1,:)*V_mbfpca, X_train(Z_train==0,:)*V_mbfpca, sigma);
        exp_vars_mbfpca_train(split, name_num) = 100 * trace(V_mbfpca'*A_train*V_mbfpca)/trace(A_train);
        % test
        mmds_mbfpca_test(split, name_num) = mmd(X(Z==1,:)*V_mbfpca, X(Z==0,:)*V_mbfpca, sigma);
        exp_vars_mbfpca_test(split, name_num) = 100 * trace(V_mbfpca'*A*V_mbfpca)/trace(A);
        
        % fairness metrics
        [acc, DP, EOP, EOD] = fairness_metric(X*V_mbfpca, Y, Z);
        accs_mbfpca(split, name_num) = acc;
        DPs_mbfpca(split, name_num) = DP;
        EOPs_mbfpca(split, name_num) = EOP;
        EODs_mbfpca(split, name_num) = EOD;
    end
end

writematrix(exp_vars_mbfpca_train, 'mbfpca/exp_vars_train.csv')
writematrix(mmds_mbfpca_train, 'mbfpca/mmds_train.csv')
writematrix(exp_vars_mbfpca_test, 'mbfpca/exp_vars_test.csv')
writematrix(mmds_mbfpca_test, 'mbfpca/mmds_test.csv')

writematrix(accs_mbfpca, 'mbfpca/accs.csv')
writematrix(DPs_mbfpca, 'mbfpca/DPs.csv')
writematrix(EOPs_mbfpca, 'mbfpca/EOPs.csv')
writematrix(EODs_mbfpca, 'mbfpca/EODs.csv')

