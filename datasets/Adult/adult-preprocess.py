"""
Created on 2021/08/25
@author: nicklee

(Description)
"""
import numpy as np
from aif360.datasets import AdultDataset

# adult = load_preproc_data_adult(protected_attributes=['sex'])
adult = AdultDataset(
    protected_attribute_names=['sex'],
    privileged_classes=[['Male']],
    features_to_drop=['fnlwgt', 'race'],
)

# Reduce the dataset to 5%...
adult, _ = adult.split([0.05], shuffle=True)
print(adult.features.shape)

# 10 different 70-30 train-test splits
for i in range(10):
    train, test = adult.split([0.7], shuffle=True)
    
    # normalize each feature, using train!
    train_mean = np.mean(train.features, axis=0)
    train_std = np.std(train.features, axis=0)
    train_std[train_std == 0] = 1

    train.features = (train.features - train_mean) / train_std
    test.features = (test.features - train_mean) / train_std

    np.savetxt('train_{}.csv'.format(i),
               np.concatenate((train.features, train.labels, train.protected_attributes), axis=1),
               delimiter=',')
    np.savetxt('test_{}.csv'.format(i),
               np.concatenate((test.features, test.labels, test.protected_attributes), axis=1),
               delimiter=',')

np.savetxt('full.csv',
           np.concatenate((adult.features, adult.labels, adult.protected_attributes), axis=1),
           delimiter=',')
# privileged_groups = [{'age': 1}]
# unprivileged_groups = [{'age': 0}]
