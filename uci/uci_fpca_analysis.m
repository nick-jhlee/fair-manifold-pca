names = {'COMPAS', 'German', 'Adult'};

mmds_fpca_train = zeros(10, 3);
exp_vars_fpca_train = zeros(10, 3);
mmds_fpca_test = zeros(10, 3);
exp_vars_fpca_test = zeros(10, 3);

accs_fpca = zeros(10, 3);
DPs_fpca = zeros(10, 3);
EOPs_fpca = zeros(10, 3);
EODs_fpca = zeros(10, 3);

d = 10;
% d = 2;

%% dimensions considered
for name_num = 2:2
% for name_num = 1:3
    for split = 1:10
        %% Load datas
        X_train = table2array(readtable(sprintf('../datasets/%s/train_%d.csv', names{name_num}, split-1)));
        Y_train = X_train(:, end-1);
        Z_train = X_train(:, end);
        X_train = X_train(:, 1:end-2);
        A_train = cov(X_train);
        
        X = table2array(readtable(sprintf('../datasets/%s/test_%d.csv', names{name_num}, split-1)));
        V_fpca = table2array(readtable(sprintf('%s/10_fpca_0.0/FPCA_V_%d.csv', names{name_num}, split-1)));
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
        mmds_fpca_train(split, name_num) = mmd(X_train(Z_train==1,:)*V_fpca, X_train(Z_train==0,:)*V_fpca, sigma);
        exp_vars_fpca_train(split, name_num) = 100 * trace(V_fpca'*A_train*V_fpca)/trace(A_train);
        % test
        mmds_fpca_test(split, name_num) = mmd(X(Z==1,:)*V_fpca, X(Z==0,:)*V_fpca, sigma);
        exp_vars_fpca_test(split, name_num) = 100 * trace(V_fpca'*A*V_fpca)/trace(A);
        
        % fairness metrics
        [acc, DP, EOP, EOD] = fairness_metric(X*V_fpca, Y, Z);
        accs_fpca(split, name_num) = acc;
        DPs_fpca(split, name_num) = DP;
        EOPs_fpca(split, name_num) = EOP;
        EODs_fpca(split, name_num) = EOD;
    end
end

% writematrix(exp_vars_fpca_train, 'fpca/exp_vars_train.csv')
% writematrix(mmds_fpca_train, 'fpca/mmds_train.csv')
% writematrix(exp_vars_fpca_test, 'fpca/exp_vars_test.csv')
% writematrix(mmds_fpca_test, 'fpca/mmds_test.csv')
% 
% writematrix(accs_fpca, 'fpca/accs.csv')
% writematrix(DPs_fpca, 'fpca/DPs.csv')
% writematrix(EOPs_fpca, 'fpca/EOPs.csv')
% writematrix(EODs_fpca, 'fpca/EODs.csv')

