function MTestSPDMean()
    seed = floor(rand() * 100000);
%     seed = 2;
    fprintf('MTestSPDMean seed:%d\n', seed);
    rand('state', seed);
    randn('state', seed);
    n = 80;
    k = 30;
    
%     As = matrix_data(n, k, 2);
    for i = 1 : k
        tmp = rand(n, n);
        As{i} = tmp' * tmp;
    end

    Ls = zeros(n, n, k);
    for i = 1 : k
        Ls(:, :, i) = chol(As{i}, 'lower');
    end
    
    Xinitial = eye(n);
    
%     SolverParams.method = 'LRBFGS';
    SolverParams.method = 'RSD';
%     SolverParams.method = 'LRTRSR1';
%     SolverParams.method = 'RTRNewton';
    SolverParams.IsCheckParams = 1;
    SolverParams.Max_Iteration = 100;
    SolverParams.LengthSY = 4;
    SolverParams.Verbose = 2;
    SolverParams.Tolerance = 1e-8;
    SolverParams.Num_pre_funs = 10;
%     SolverParams.Accuracy = 1e-5;
%     SolverParams.PreFunsAccuracy = 1e-4;
    HasHHR = 0;
    ParamSet = 1;
    [Xopt, f, gf, gfgf0, iter, nf, ng, nR, nV, nVp, nH, ComTime] = TestSPDKarcherMean(Ls, Xinitial, HasHHR, ParamSet, SolverParams);
end

%% generate test data set
function As = matrix_data(n, k, data_type)
    As = cell(k,1);
    if(data_type == 1) %ill
        CN = 4;
        for i = 1 : CN
            [O,~] = qr(randn(n));
            f = 0;
            D = diag([[rand(1,n-1)+1],10^(-f)]);
            As{i} = O * D * O';
        end
        for i = (CN+1) : k
            [O,~] = qr(randn(n));
            f = 5;
            %D = 100000 * diag([[rand(1,n-1)+1],10^(-f)]);
            D = diag([[rand(1,n-1)+1],10^(-f)]);
            As{i} = O * D * O';
            % aha = cond(As{i})
        end
    end


    if(data_type == 3)
        [O,~] = qr(randn(n));
        f = 0;
        D = diag([[rand(1,n-1)+1],10^(-f)]);
        As{1} = O * D * O';

        for i = 2 : k
            [O,~] = qr(randn(n));
            f = 5;
            m = 70;
            Dx = diag([rand(1,m)+1, (rand(1,n-m)+1)*10^(f)]);
            ill = O * Dx * O';
            As{i} = As{1} * ill * As{1};
            % aha = cond(As{i})
        end

    end



    if(data_type == 2)%structured well
        grp = floor(k/3);
        for i = 1 : grp
            f = 0;
            [O,~] = qr(randn(n));
            D = diag([[rand(1,n-1)+1],10^(-f)]);
            As{i} = O * D * O';
        end
        for i = (grp+1) : (2*grp)
            f = 1;
            [O,~] = qr(randn(n));
            D = diag([[rand(1,n-1)+1],10^(-f)]);
            As{i} = O * D * O';
        end
        for i = (2*grp + 1) : k
            f = 2;
            [O,~] = qr(randn(n));
            D = diag([[rand(1,n-1)+1],10^(-f)]);
            As{i} = O * D * O';
        end
    end
end
