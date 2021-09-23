"""
Created on 2021/08/25
@author: nicklee

(Description)
"""
import numpy as np
from aif360.datasets import CompasDataset

# compas = load_preproc_data_compas(protected_attributes=['race'])
compas = CompasDataset(
    protected_attribute_names=['race'],
    privileged_classes=[['Caucasian']],
    features_to_drop=['sex', 'c_charge_desc'],
)

# Reduce the dataset to 40%...
compas, _ = compas.split([0.4], shuffle=True)
print(compas.features.shape)

# 10 different 70-30 train-test splits
for i in range(10):
    train, test = compas.split([0.7], shuffle=True)

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
           np.concatenate((compas.features, compas.labels, compas.protected_attributes), axis=1),
           delimiter=',')