import tensorflow as tf

print(tf.__version__)

from sklearn.model_selection import KFold
import numpy as np

x = np.array([[10, 20], [20, 30], [30, 40], [40, 50], [50, 60]])
y = np.array([100, 200, 300, 400, 500])

kf = KFold(n_splits= 2, random_state= 2018, shuffle= True)
for fold, (train_index, valid_index) in enumerate(kf.split(y)):
    print(fold)
    print(train_index, valid_index)
