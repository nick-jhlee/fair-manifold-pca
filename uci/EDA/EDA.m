races = zeros(10, 3);
vars = zeros(10, 3);

% name = 'COMPAS';
% name = 'German';
name = 'Adult';

protected_num = 3;
protected_name = 'Gender';

d = 2;

lim = [-0.5, 0.5];

plot_biplot = true;

for split=0:9
    V_fpca = table2array(readtable(sprintf('../%s/%d_fpca_0.0/FPCA_V_%d.csv', name, d, split)));
    V_stfpca = table2array(readtable(sprintf('../%s/%d_stfpca_6/STFPCA_V_%d.csv', name, d, split)));

    %% Load datas
    X_train = table2array(readtable(sprintf('../../datasets/%s/train_%d.csv', name, split)));
    Z_train = X_train(:, end);
    X_train = X_train(:, 1:end-2);
    A_train = cov(X_train);

    % Obtain PCA
    V_pca = pca(X_train);
    V_pca = V_pca(:, 1:d);


    %% Plot
%     variables = {'age', 'race', 'juv_fel_count', 'juv_misd_count', 'juv_other_count', 'priors_count', 'age_cat=25 - 45', 'age_cat=Greater than 45', 'age_cat=Less than 25', 'c_charge_degree=F', 'c_charge_degree=M'};
%     variables = {'month', 'credit_amount', 'investment_as_income_percentage', 'residence_since', 'age', 'number_of_credits', 'people_liable_for', 'status=A11', 'status=A12', 'status=A13', 'status=A14', 'credit_history=A30', 'credit_history=A31', 'credit_history=A32', 'credit_history=A33', 'credit_history=A34', 'purpose=A40', 'purpose=A41', 'purpose=A410', 'purpose=A42', 'purpose=A43', 'purpose=A44', 'purpose=A45', 'purpose=A46', 'purpose=A48', 'purpose=A49', 'savings=A61', 'savings=A62', 'savings=A63', 'savings=A64', 'savings=A65', 'employment=A71', 'employment=A72', 'employment=A73', 'employment=A74', 'employment=A75', 'other_debtors=A101', 'other_debtors=A102', 'other_debtors=A103', 'property=A121', 'property=A122', 'property=A123', 'property=A124', 'installment_plans=A141', 'installment_plans=A142', 'installment_plans=A143', 'housing=A151', 'housing=A152', 'housing=A153', 'skill_level=A171', 'skill_level=A172', 'skill_level=A173', 'skill_level=A174', 'telephone=A191', 'telephone=A192', 'foreign_worker=A201', 'foreign_worker=A202'};
    variables = ...
        {'age', 'education-num', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'workclass=Federal-gov', 'workclass=Local-gov', 'workclass=Private', 'workclass=Self-emp-inc', 'workclass=Self-emp-not-inc', 'workclass=State-gov', 'workclass=Without-pay', 'education=10th', 'education=11th', 'education=12th', 'education=1st-4th', 'education=5th-6th', 'education=7th-8th', 'education=9th', 'education=Assoc-acdm', 'education=Assoc-voc', 'education=Bachelors', 'education=Doctorate', 'education=HS-grad', 'education=Masters', 'education=Preschool', 'education=Prof-school', 'education=Some-college', 'marital-status=Divorced', 'marital-status=Married-AF-spouse', 'marital-status=Married-civ-spouse', 'marital-status=Married-spouse-absent', 'marital-status=Never-married', 'marital-status=Separated', 'marital-status=Widowed', 'occupation=Adm-clerical', 'occupation=Armed-Forces', 'occupation=Craft-repair', 'occupation=Exec-managerial', 'occupation=Farming-fishing', 'occupation=Handlers-cleaners', 'occupation=Machine-op-inspct', 'occupation=Other-service', 'occupation=Priv-house-serv', 'occupation=Prof-specialty', 'occupation=Protective-serv', 'occupation=Sales', 'occupation=Tech-support', 'occupation=Transport-moving', 'relationship=Husband', 'relationship=Not-in-family', 'relationship=Other-relative', 'relationship=Own-child', 'relationship=Unmarried', 'relationship=Wife', 'native-country=Cambodia', 'native-country=Canada', 'native-country=China', 'native-country=Columbia', 'native-country=Cuba', 'native-country=Dominican-Republic', 'native-country=Ecuador', 'native-country=El-Salvador', 'native-country=England', 'native-country=France', 'native-country=Germany', 'native-country=Greece', 'native-country=Guatemala', 'native-country=Haiti', 'native-country=Holand-Netherlands', 'native-country=Honduras', 'native-country=Hong', 'native-country=Hungary', 'native-country=India', 'native-country=Iran', 'native-country=Ireland', 'native-country=Italy', 'native-country=Jamaica', 'native-country=Japan', 'native-country=Laos', 'native-country=Mexico', 'native-country=Nicaragua', 'native-country=Outlying-US(Guam-USVI-etc)', 'native-country=Peru', 'native-country=Philippines', 'native-country=Poland', 'native-country=Portugal', 'native-country=Puerto-Rico', 'native-country=Scotland', 'native-country=South', 'native-country=Taiwan', 'native-country=Thailand', 'native-country=Trinadad&Tobago', 'native-country=United-States', 'native-country=Vietnam', 'native-country=Yugoslavia'};

    X1 = X_train(Z_train==1, :);
    X2 = X_train(Z_train==0, :);
    % 
    % figure(1)
    % X = categorical(variables);
    % X = reordercats(X,variables);
    communalities_pca = vecnorm(V_pca') .^ 2;
    communalities_fpca = vecnorm(V_fpca') .^ 2;
    communalities_stfpca = vecnorm(V_stfpca') .^ 2;
    
    races(split+1, 1) = communalities_pca(protected_num);
    races(split+1, 2) = communalities_fpca(protected_num);
    races(split+1, 3) = communalities_stfpca(protected_num);
    % bar(X, [communalities_pca; communalities_fpca; communalities_stfpca]')
    % legend('PCA', 'FPCA', 'St-FPCA')
    
    vars(split+1, 1) = 100*trace(V_pca'*A_train*V_pca)/trace(A_train);
    vars(split+1, 2) = 100*trace(V_fpca'*A_train*V_fpca)/trace(A_train);
    vars(split+1, 3) = 100*trace(V_stfpca'*A_train*V_stfpca)/trace(A_train);
end

%% Plot
split = 5;

V_fpca = table2array(readtable(sprintf('../%s/%d_fpca_0.0/FPCA_V_%d.csv', name, d, split)));
V_stfpca = table2array(readtable(sprintf('../%s/%d_stfpca_6/STFPCA_V_%d.csv', name, d, split)));
%% Load datas
X_train = table2array(readtable(sprintf('../../datasets/%s/train_%d.csv', name, split)));
Y_train = X_train(:, end-1);
Z_train = X_train(:, end);
X_train = X_train(:, 1:end-2);
A_train = cov(X_train);

%% Pick the top 2 PCs
tmp = diag(V_fpca' * A_train * V_fpca);
[~, top_i] = maxk(tmp, 2);
V_fpca = V_fpca(:, top_i);

tmp = diag(V_stfpca' * A_train * V_stfpca);
[~, top_i] = maxk(tmp, 2);
V_stfpca = V_stfpca(:, top_i);


% Obtain PCA
V_pca = pca(X_train);
V_pca = V_pca(:, 1:2);

%% Pick the top 10 features based on their communality
communalities_pca = vecnorm(V_pca');
communalities_fpca = vecnorm(V_fpca');
communalities_stfpca = vecnorm(V_stfpca');

[~, top_pca] = maxk(communalities_pca, 10);
[~, top_fpca] = maxk(communalities_fpca, 10);
[~, top_stfpca] = maxk(communalities_stfpca, 10);

if ~ismember(protected_num, top_pca)
    top_pca = [top_pca protected_num];
end

if ~ismember(protected_num, top_fpca)
    top_fpca = [top_fpca protected_num];
end

if ~ismember(protected_num, top_stfpca)
    top_stfpca = [top_stfpca protected_num];
end

V_pca = V_pca(top_pca, :);
V_fpca = V_fpca(top_fpca, :);
V_stfpca = V_stfpca(top_stfpca, :);

%% Plot
figure(1)
bar(races)

figure(101)
bh = boxplot(races, {'PCA', 'FPCA', 'MbF-PCA'});
lgd = xlabel('Algorithm');
lgd.FontSize = 40;
rgd = ylabel(sprintf('Communality of %s', protected_name));
rgd.FontSize = 40;
set(gca,'FontSize',26);
set(bh,'LineWidth', 2);
% 
% 
% figure(100)
% bar(vars)
% title('Total explained variance for each split')


if plot_biplot
    close all;
    set(0, 'defaultTextFontSize', 10)
    figure(2)
    p1 = axes();
    biplot(p1, V_pca, 'VarLabels', variables(top_pca))
    xlim(p1, lim)
    ylim(p1, lim)
    % hold on
    % tmp = X1*V_pca;
    % tmp_ = X2*V_pca;
    % scatter(tmp(:,1), tmp(:,2));
    % scatter(tmp_(:,1), tmp_(:,2));
    % hold off

    figure(3)
    p2 = axes();
    biplot(p2, V_fpca, 'VarLabels', variables(top_fpca))
    xlim(p2, lim)
    ylim(p2, lim)
    % hold on
    % tmp = X1*V_fpca;
    % tmp_ = X2*V_fpca;
    % scatter(tmp(:,1), tmp(:,2));
    % scatter(tmp_(:,1), tmp_(:,2));
    % hold off

    figure(4)
    p3 = axes();
    biplot(p3, V_stfpca, 'VarLabels', variables(top_stfpca))
    xlim(p3, lim)
    ylim(p3, lim)
    % hold on
    % tmp = X1*V_stfpca;
    % tmp_ = X2*V_stfpca;
    % scatter(tmp(:,1), tmp(:,2));
    % scatter(tmp_(:,1), tmp_(:,2));
    % hold off

    % figure(5)
    % plotmatrix(X_train*V_pca)
    % 
    % figure(6)
    % plotmatrix(X_train*V_fpca)
    % 
    % figure(7)
    % plotmatrix(X_train*V_stfpca)
    % 
    % %% Communalities w.r.t. race and age and age>=45
    % communalities_pca = vecnorm(V_pca') .^ 2;
    % communalities_fpca = vecnorm(V_fpca') .^ 2;
    % communalities_stfpca = vecnorm(V_stfpca') .^ 2;
    % 
    % results = zeros(3, 3);
    % 
    % results(:, 1) = communalities_pca([1 2 8])';
    % results(:, 2) = communalities_fpca([1 2 8])';
    % results(:, 3) = communalities_stfpca([1 2 8])';
    % 
    % figure(101)
    % names = {'age', 'race', 'age >= 45'};
    % X = categorical(names);
    % X = reordercats(X, names);
    % bar(X, results);
    % legend('PCA', 'FPCA', 'MbF-PCA')
end


