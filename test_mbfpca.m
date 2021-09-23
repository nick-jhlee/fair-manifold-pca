% MyMex DriverMexProb;    % Run only once!
% clear;
%% User-defined parameters
d = 3;
p = 115;
q = p/5;
fairness = 'DEM';
generate = true;


%% Create unfair synthetic data
if generate
%     rng(17)
    n1 = 5000;
    n2 = 5000;

    mu1 = zeros(p, 1);
    mu2 = [ones(2*q, 1); zeros(3*q, 1)];

    sigma1 = blkdiag(ones(q, q), 0.5*ones(q, q), 0.1*ones(q, q), ...
        0.5*ones(q, q), ones(q, q));
%     sigma2 = sigma1;
    sigma2 = blkdiag(0.5*ones(q, q), ones(q, q), 0.11*ones(q, q), ...
        0.5*ones(q, q), ones(q, q));

    X1 = mvnrnd(mu1, sigma1, n1);
    X2 = mvnrnd(mu2, sigma2, n2);

    X = [X1; X2];
    X = normalize(X);

    X1 = X(1:n1, :);
    x2 = X(n1+1:end, :);

    Y = randn(n1+n2, 1) > 0;
    % Z = (Y .* randn(1000, 1)) > 0;
    % Z = 2*Z - 1;
    Z = [ones(n1, 1); zeros(n2, 1)];

    X = [X Y Z];
else
    X = table2array(readtable('X.csv'));
    
    X(:, 1:end-2) = normalize(X(:, 1:end-2));
    
    X(:, end-1) = 2*X(:, end-1) - 1;
    Y = X(:, end-1);
    
    X(:, end) = 2*X(:, end) - 1;
    Z = X(:, end);
end

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

%% Run MbF-PCA
% [V, logs_] = mbfpca_auxiliary(X, d, fairness, sigma, rho0);
tau = 1e-5;
[V, logs] = mbfpca(X, d, fairness, sigma, rho0, tau);
V = V.main;

X = X(:, 1:end-2);

%% Compare it with PCA
disp('Fair PCA')
mmd_DP = mmd(X(Z == 1,:)*V, X(Z == 0,:)*V, sigma)
mmd_EOP = mmd(X((Y == 1) & (Z == 1),:)*V, X((Y == 1) & (Z == 0),:)*V, sigma)
mmd_EOD = max(mmd_EOP, mmd(X((Y == 0) & (Z == 1),:)*V, X((Y == 0) & (Z == 0),:)*V, sigma))

disp('Vanilla PCA')
mmd_DP_pca = mmd(X(Z == 1,:)*V_pca, X(Z == 0,:)*V_pca, sigma)
mmd_EOP_pca = mmd(X((Y == 1) & (Z == 1),:)*V_pca, X((Y == 1) & (Z == 0),:)*V_pca, sigma)
mmd_EOD_pca = max(mmd_EOP, mmd(X((Y == 0) & (Z == 1),:)*V_pca, X((Y == 0) & (Z == 0),:)*V_pca, sigma))


%% Plot
figure(1)
X_2 = X(n1+1:end,:) * V;
scatter3(X_2(:,1), X_2(:,2), X_2(:,3));
hold on
X_ = X(1:n1,:) * V;
scatter3(X_(:,1), X_(:,2), X_(:,3));
hold off

SVMModel = fitcsvm(X*V, Z);
[label, ~] = predict(SVMModel, X*V);
sum(label == Z) / size(Z,1)


figure(2)
X_ = X(1:n1,:) * V_pca;
scatter3(X_(:,1), X_(:,2), X_(:,3));
hold on
X_2 = X(n1+1:end,:) * V_pca;
scatter3(X_2(:,1), X_2(:,2), X_2(:,3));

SVMModel = fitcsvm(X*V_pca, Z);
[label, ~] = predict(SVMModel, X*V_pca);
sum(label == Z) / size(Z,1)

hold off
% 
% 
% figure(3)
% heatmap(abs(V))
% 
% figure(4)
% heatmap(abs(V_pca))