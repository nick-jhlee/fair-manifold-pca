d = 2;
fairness = 'DP';

X = table2array(readtable('synthetic1.csv'));
Z = X(:, end);

n1 = sum(Z);
n2 = sum(Z == 0);


%% Run PCA
V_pca = pca(X(:,1:end-2));
V_pca = V_pca(:, 1:d);

% Median heuristic for bandwidth choice (Scholkopf and Smola, 2002)
sigma = sqrt(median(pdist(X(:,1:end-2)*V_pca, 'squaredeuclidean'))/2);
% sigma = 0.01;
% sigma = sqrt(median(pdist(X(:,1:end-2), 'squaredeuclidean')));
X1 = X(Z == 1, 1:end-2);
X2 = X(Z == 0, 1:end-2);
m_ = mmd(X1*V_pca, X2*V_pca, sigma)
rho0 = 0.1/m_;
tau = 1e-5;

%% Run MbF-PCA
% [V, logs_] = mbfpca_auxiliary(X, d, fairness);
[V, logs] = mbfpca(X, d, fairness, sigma, rho0, tau);
V = V.main;

X = X(:, 1:end-2);

%% Compare it with PCA
disp('Fair PCA')
mmd_DP = mmd(X(Z == 1,:)*V, X(Z == 0,:)*V, sigma)

disp('Vanilla PCA')
mmd_DP_pca = mmd(X(Z == 1,:)*V_pca, X(Z == 0,:)*V_pca, sigma)


%% Plot
figure(1)
X_ = X(1:n1,:);
X_2 = X(n1+1:end,:);
scatter3(X_2(:,1), X_2(:,2), X_2(:,3));
hold on
scatter3(X_(:,1), X_(:,2), X_(:,3));
hold off


figure(2)
X_ = X(1:n1,:) * V;
X_2 = X(n1+1:end,:) * V;
scatter(X_2(:,1), X_2(:,2));
hold on
scatter(X_(:,1), X_(:,2));
hold off

SVMModel = fitcsvm(X*V, Z);
[label, ~] = predict(SVMModel, X*V);
sum(label == Z) / size(Z,1)


figure(3)
X_ = X(1:n1,:) * V_pca;
X_2 = X(n1+1:end,:) * V_pca;
scatter(X_2(:,1), X_2(:,2));
hold on
scatter(X_(:,1), X_(:,2));
hold off

SVMModel = fitcsvm(X*V_pca, Z);
[label, ~] = predict(SVMModel, X*V_pca);
sum(label == Z) / size(Z,1)
%
% 
% figure(3)
% heatmap(abs(V))
% 
% figure(4)
% heatmap(abs(V_pca))