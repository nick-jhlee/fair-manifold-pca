% MMD^2
function d = mmd(X1, X2, sigma)
    m = size(X1, 1);
    n = size(X2, 1);
    
    d = (1/m^2) * ones(1,m) * rbf(X1, X1, sigma) * ones(m,1);
    d = d + (1/n^2) * ones(1,n) * rbf(X2, X2, sigma) * ones(n,1);
    d = d - (2/(m*n)) * ones(1,m) * rbf(X1, X2, sigma) * ones(n,1);
end

function K = rbf(X1, X2, sigma)
    K = pdist2(X1, X2, 'squaredeuclidean');
    K = exp(-(1/(2*sigma^2)) * K);
%     if clear_diag
%         K = K - diag(diag(K));
%     end
end