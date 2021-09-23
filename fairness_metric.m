function [acc, DP, EOP, EOD] = fairness_metric(X_test, Y_test, Z_test)
    % Train RBF SVM
%     SVMModel = fitcsvm(X_train, Y_train);
%     SVMModel = fitcsvm(X_train, Y_train, 'KernelFunction', 'RBF');
    SVMModel = fitcsvm(X_test, Y_test, 'KernelFunction', 'RBF');
%     [label, ~] = predict(SVMModel, X_);
    
    % Obtain labels of test data
    [label, ~] = predict(SVMModel, X_test);
    
    %% downstream accuracy
    acc = (1/size(X_test, 1))*sum(label == Y_test);
    
    %% fairness metrics
    % Split datas
    X1_ = X_test(Z_test == 1, :);
    X2_ = X_test(Z_test == 0, :);
    
    % DP
%     sprintf('|%d - %d|', (1/size(X1_, 1))*sum(label(Z_test == 1)), (1/size(X2_, 1))*sum(label(Z_test == 0)))
    DP = abs((1/size(X1_, 1))*sum(label(Z_test == 1)) - (1/size(X2_, 1))*sum(label(Z_test == 0)));
    
    % EOP
    EOP = abs((1/size(X1_, 1))*sum(label((Y_test == 1) & (Z_test == 1))) -...
        (1/size(X2_, 1))*sum(label((Y_test == 1) & (Z_test == 0))));
    
    % EOD
    EOD = max(EOP, ...
        abs((1/size(X1_, 1))*sum(label(Z_test == 1)) -...
        (1/size(X2_, 1))*sum(label(Z_test == 0))));
end