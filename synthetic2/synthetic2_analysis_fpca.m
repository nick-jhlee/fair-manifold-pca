mmds_fpca_train = zeros(10, 9);
exp_vars_fpca_train = zeros(10, 9);
mmds_fpca_test = zeros(10, 9);
exp_vars_fpca_test = zeros(10, 9);

d = 5;

tmp = 1;
%% dimensions considered
for p = 20:10:100
    for split = 1:10
        %% Load datas
        X_train = table2array(readtable(sprintf('%d/train_%d.csv', p, split)));
        X_train = X_train(:, 1:end-2);
        A_train = cov(X_train);
        
        X = table2array(readtable(sprintf('%d/test_%d.csv', p, split)));
        V_fpca = table2array(readtable(sprintf('%d/FPCA_V_%d.csv', p, split)));
        
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
        mmds_fpca_train(split, tmp) = mmd(X_train(Z==1,:)*V_fpca, X_train(Z==0,:)*V_fpca, sigma);
        exp_vars_fpca_train(split, tmp) = 100 * trace(V_fpca'*A_train*V_fpca)/trace(A_train);
        % test
        mmds_fpca_test(split, tmp) = mmd(X(Z==1,:)*V_fpca, X(Z==0,:)*V_fpca, sigma);
        exp_vars_fpca_test(split, tmp) = 100 * trace(V_fpca'*A*V_fpca)/trace(A);
    end
    
    tmp = tmp + 1;
end

writematrix(exp_vars_fpca_train, 'fpca/exp_vars_train.csv')
writematrix(mmds_fpca_train, 'fpca/mmds_train.csv')
writematrix(exp_vars_fpca_test, 'fpca/exp_vars_test.csv')
writematrix(mmds_fpca_test, 'fpca/mmds_test.csv')
