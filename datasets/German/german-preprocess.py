"""
Created on 2021/08/25
@author: nicklee

(Description)
"""
import numpy as np
from aif360.datasets import GermanDataset

# german = load_preproc_data_german(protected_attributes=['age'])
german = GermanDataset(
    protected_attribute_names=['age'],
    privileged_classes=[lambda x: x >= 25],      # age >=25 is considered privileged
    features_to_drop=['sex', 'personal_status']
)
print(german.features.shape)

# 10 different 70-30 train-test splits
for i in range(10):
    train, test = german.split([0.7], shuffle=True)

    # normalize each feature, using train!
    train_mean = np.mean(train.features, axis=0)
    train_std = np.std(train.features, axis=0)
    train_std[train_std == 0] = 1

    train.features = (train.features - train_mean) / train_std
    test.features = (test.features - train_mean) / train_std

    np.savetxt('train_{}.csv'.format(i),
               np.concatenate((train.features, train.labels - 1, train.protected_attributes), axis=1),
               delimiter=',')
    np.savetxt('test_{}.csv'.format(i),
               np.concatenate((test.features, test.labels - 1, test.protected_attributes), axis=1),
               delimiter=',')

np.savetxt('full.csv',
           np.concatenate((german.features, german.labels - 1, german.protected_attributes), axis=1),
           delimiter=',')
# privileged_groups = [{'age': 1}]
# unprivileged_groups = [{'age': 0}]
