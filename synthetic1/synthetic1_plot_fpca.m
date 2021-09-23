X = table2array(readtable('synthetic1.csv'));
Z = X(:, end);
X = X(:, 1:end-2);
V = table2array(readtable('V_synthetic1.csv'));

n1 = sum(Z);
n2 = sum(Z == 0);


V_pca = pca(X);
V_pca = V_pca(:, 1:d);
sigma = sqrt(median(pdist(X*V_pca, 'squaredeuclidean'))/2);


%% Compare it with PCA
disp('FPCA')
mmd_DP_fpca = mmd(X(Z == 1,:)*V, X(Z == 0,:)*V, sigma)


%% Plot
figure(4)
X_ = X(1:n1,:) * V;
X_2 = X(n1+1:end,:) * V;
scatter(X_2(:,1), X_2(:,2));
hold on
scatter(X_(:,1), X_(:,2));
hold off

SVMModel = fitcsvm(X*V, Z);
[label,score] = predict(SVMModel, X*V);
sum(label == Z) / size(Z,1)
