rng default

n = 240;

n1 = 168;
n2 = 72;

% for p = 10:10:100
% for p = 200:100:1000
% for p = 6000:1000:10000
p0 = 1000;
for p = [20:10:100 200:100:1000]
    %% Make directory
    mkdir(sprintf('%d', p));
    
    %% Means
    mu1 = zeros(p0, 1);
    % Mean difference, normalized!
%     f = [ones(3*p/5, 1); zeros(2*p/5, 1)];
%     f = ones(p0, 1);
    f = [ones(p0/5, 1); ones(3*p0/5, 1); ones(p0/5, 1)];
    f = 2 * f / norm(f);
    mu2 = mu1 + f;
    
    %% Covariances
%     Sigma1 = blkdiag(AR(0.99, p0/5), AR(0.98, p0/5), AR(0.97, p0/5), AR(0.96, p0/5), AR(0.95, p0/5));
%     Sigma2 = blkdiag(AR(0.95, p0/5), AR(0.96, p0/5), AR(0.97, p0/5), AR(0.98, p0/5), AR(0.99, p0/5));
    Sigma1 = blkdiag(AR(0.99, p0/5), AR(0.98, p0/5), AR(0.97, p0/5), AR(0.98, p0/5), AR(0.95, p0/5));
    Sigma2 = blkdiag(AR(0.95, p0/5), AR(0.98, p0/5), AR(0.97, p0/5), AR(0.98, p0/5), AR(0.99, p0/5));
    
    % Covariance difference, normalized!
    Q = Sigma2 - Sigma1;
    Sigma1 = Sigma1 / norm(Q);
    Sigma2 = Sigma2 / norm(Q);
    Q = Sigma2 - Sigma1;
    
    %% Sample/save datas
    X1 = mvnrnd(mu1, Sigma1, n) * randn(p0, p);
    X2 = mvnrnd(mu2, Sigma2, n) * randn(p0, p);
%     tmp = randn(p0, p);
%     X1 = mvnrnd(mu1, Sigma1, n) * tmp;
%     X2 = mvnrnd(mu2, Sigma2, n) * tmp;
    
    for split=1:10
        cv = cvpartition(n, 'HoldOut', 0.3);
        
        z1 = [ones(n1, 1); zeros(n1, 1)];
        X1_train = X1(cv.training, :);
        X2_train = X2(cv.training, :);
        X_train = [X1_train; X2_train];
        writematrix([X_train z1 z1], sprintf('%d/train_%d.csv', p, split));

        z2 = [ones(n2, 1); zeros(n2, 1)];
        X1_test = X1(cv.test, :);
        X2_test = X2(cv.test, :);
        X_test = [X1_test; X2_test];
        writematrix([X_test z2 z2], sprintf('%d/test_%d.csv', p, split));
    end
%     
%     %% Plot
%     V = pca(X);
%     V = V(:, 1:5);
%     X1_ = X1 * V;
%     X2_ = X2 * V;
%     
%     figure(p)
%     scatter3(X1_(:, 1), X1_(:, 2), X1_(:, 3), 'blue')
%     hold on
%     scatter3(X2_(:, 1), X2_(:, 2), X2_(:, 3), 'red')
%     hold off
end

function Sigma = AR(r, p)
    Sigma = zeros(p, p);
    for i = 1:p
        for j = 1:p
            Sigma(i, j) = r^abs(i - j) / (1 - r^2);
        end
    end
end