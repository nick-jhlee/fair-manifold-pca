n1 = 5000;

mean1 = -ones(1,3);
cov1 = 0.1*eye(3);

mean2 = ones(1,3);
cov2 = 0.1*eye(3);


X1 = mvnrnd(mean1, cov1, n1/2);
X2 = mvnrnd(mean2, cov2, n1/2);
X = [X1; X2];


scatter3(X(:,1), X(:,2), X(:,3))


n2 = 5000;
mean3 = zeros(1,3);
cov3 = 0.1*eye(3) + ones(3);
X_ = mvnrnd(mean3, cov3, n2);
hold on
scatter3(X_(:,1), X_(:,2), X_(:,3))
hold off

Z = [ones(n1, 1); zeros(n2, 1)];

X = [X; X_];
X = [X Z Z];

% Save as csv
csvwrite('synthetic1.csv', X);