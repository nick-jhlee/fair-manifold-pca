exp_vars_fpca = zeros(10, 57);
exp_vars_stfpca = zeros(10, 57);
name = 'German';


for split=0:9
    X_test = table2array(readtable(sprintf('../../datasets/%s/test_%d.csv', name, split)));
    X_test = X_test(:, 1:end-2);
    A_test = cov(X_test);
    
    P_fpca = table2array(readtable(sprintf('FPCA_P_%d.csv', split)));    
    exp_vars_fpca(split+1, :) = 100 * diag(P_fpca' * A_test * P_fpca) / trace(A_test);

    V_stfpca = table2array(readtable(sprintf('../%s/10_stfpca_3/STFPCA_V_%d.csv', name, split)));    
    exp_vars_stfpca(split+1, :) = [100 * diag(V_stfpca' * A_test * V_stfpca) / trace(A_test); zeros(47, 1)]';
end

%% Plot
% figure(1)
% bh = boxplot(flip(exp_vars_train,2))
% lgd = xlabel('Each dimension of P');
% lgd.FontSize = 30;
% rgd = ylabel('Explained variance(%)');
% rgd.FontSize = 30;
% hold on
% xline(10.5,'-', {'Cutoff for final loading matrix'}, 'LabelOrientation', 'horizontal', 'LineWidth', 3, 'Color', 'r');
% hold off
% set(gca,'FontSize', 6);
% set(bh,'LineWidth', 1);
% print('german_train', '-dpdf', '-bestfit')

figure(1)
bh = boxplot(flip(exp_vars_fpca,2))
xticks([1 5 10 15 20 25 30 35 40 45 50 57])
xticklabels({'1', '5', '10', '15', '20', '25', '30', '35', '40', '45', '50', '57'})
set(gca, 'FontSize', 15)
xlabel('Each dimension of P', 'FontSize', 20);
ylabel('Explained variance(%)', 'FontSize', 20);
hold on
hxl = xline(10.5,'-', {'Cutoff for final loading matrix'}, 'LabelOrientation', 'horizontal', 'LineWidth', 3, 'Color', 'r');
hxl.FontSize = 15;
hold off
% set(gca,'FontSize', 6);
set(bh,'LineWidth', 1);
% print('german_fpca', '-dpdf', '-fillpage')


figure(2)
bh = boxplot(exp_vars_stfpca)
xticks([1 5 10 15 20 25 30 35 40 45 50 57])
xticklabels({'1', '5', '10', '15', '20', '25', '30', '35', '40', '45', '50', '57'})
set(gca, 'FontSize', 15)
xlabel('Each dimension of P', 'FontSize', 20);
ylabel('Explained variance(%)', 'FontSize', 20);
hold on
hxl = xline(10.5,'-', {'Cutoff for final loading matrix'}, 'LabelOrientation', 'horizontal', 'LineWidth', 3, 'Color', 'r');
hxl.FontSize = 15;
hold off
set(bh,'LineWidth', 1);
% print('german_stfpca', '-dpdf', '-fillpage')
