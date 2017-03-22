
# apply PCA to reduce the data to 2D

import numpy as np
import pandas as pd
import matplotlib. pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import random

digits_label=pd.read_csv("./digits_label_random_sample.csv",index_col = 0)
digits = pd.read_csv("./digits_random_sample.csv", index_col = 0)

digits2 = StandardScaler(with_std=False).fit_transform(digits)
pca = PCA(n_components=2)
pca.fit(digits2)
digits_pca = pca.fit_transform(digits2)
pca.explained_variance_ratio_.sum() # get how much variance the 2D PCA can explain

#Projection to 2D

digits_transposed = np.transpose(digits)
labels_transpose = digits.columns.delete(-1)
digits_transposed_normalized = StandardScaler(with_std=False).fit_transform(digits_transposed)

pca.fit(digits_transposed_normalized)
digits_transpose_pca = pca.fit_transform(digits_transposed_normalized)

X = digits_transposed_normalized
print ("dimension of data = ", X.shape)
XTX = np.dot (X.T, X)
(sigma, U) = np.linalg.eig(XTX)

# The eigenvalues

idx = sigma.argsort()[::-1]
sigma = sigma[idx] # we sort so that largest eigens come first
U = U[: idx]

print(sum (sigma [0:2]))
print(sum(sigma))
print ("first two eigenv account for ", sum(sigma[0:2])/sum(sigma))

# Projection matrix using the first two eigenvectors

P = U [: :2]
print("eigevectors = ", U , "P = " , P)

# After projection

pca_manual = np.dot (X, P)
print ("Project data to 2D:", pca_manual)






