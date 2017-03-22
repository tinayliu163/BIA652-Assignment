# Naive Bayes
# spine dataset
# KNN 

import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import statsmodels.api as sm

from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier


# import data

sp = pd.read_csv("./Dataset_spine.csv")
sp[:4]

# convert categorical value to 1/0
def f(x):
    if "Normal" in x:
        return 0
    else:
        return 1
    
labels = sp['Class_att'].apply(lambda x: f(x))
labels

# fit Naive Bayes model

features = sp[sp.columns[0:12]]
n = features.shape[0]
sto = int(0.6*n)
data_train = features.sample(n=sto, random_state = 132151)
indices = data_train.index
label_train = labels[indices]
data_test = features[~features.index.isin(indices)]
label_test = labels[~features.index.isin(indices)]

gnb = GaussianNB()
gnb.fit(data_train, label_train)

# performance

fpr_NB, tpr_NB, thresholds_NB = metrics.roc_curve(label_test, gnb.predict_proba(data_test)[:,1], pos_label=1)
AUC = metrics.auc(fpr_NB, tpr_NB)

plt.plot(fpr_NB, tpr_NB, color="green" , label = 'ROC curve (area = %0.3f)' % AUC,)
plt.plot([0, 1], [0, 1],  color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.legend(loc="lower right")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Naive Bayes: ROC Curve ')
plt.show()

print("Naive Bayes AUC :", AUC)

# KNN model

kNN = KNeighborsClassifier(n_neighbors=6)
kNN.fit(data_train, label_train) 

# performance


fpr_kNN, tpr_kNN, thresholds_kNN = metrics.roc_curve(label_test, kNN.predict_proba(data_test)[:,1], pos_label=1)
AUC_KNN = metrics.auc(fpr_kNN, tpr_kNN)

plt.plot(fpr_kNN, tpr_kNN, color="green", label = 'ROC curve (area = %0.3f)' % AUC_KNN,)
plt.plot([0, 1], [0, 1],  color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.legend(loc="lower right")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Naive Bayes: ROC Curve ')
plt.show()

print("kNN AUC:", AUC_KNN)






