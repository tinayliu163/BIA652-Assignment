# Logistic regression 
# college admission data

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import statsmodels.api as sm
from sklearn import cross_validation
from sklearn import metrics
%matplotlib inline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix


c_ad = pd.read_csv("http://www.ats.ucla.edu/stat/data/binary.csv")

# summarize the data
print(c_ad.describe())

labels = c_ad['admit'].	values
features = c_ad[[ "gre","gpa","rank" ]]
ones=np.transpose([np.repeat(1,features.shape[0])])
features2 = np.concatenate((ones, features), axis = 1)

# split data into 60% training and 40% testing
x_train, x_test, y_train, y_test = cross_validation.train_test_split(features2, labels, test_size = 0.4, random_state = 123)


# fit logistic regression model using statsmodels

model = sm.Logit(y_train, x_train)
mymodel	= model.fit()
model_summary = mymodel.summary()
print(model_summary)

# fit logistic regression using scikit-learn

logreg = LogisticRegression()
X = x_train
Y = y_train
logreg.fit(X, Y)

# Evaluate model on 40% testing data

y_pred = logreg.predict(x_test)

# test error rate

accuracy = metrics.accuracy_score(y_test, y_pred)
error_rate = 1 - accuracy
print("test error rate : %0.3f" % error_rate)

# confusion matrix
matrix = metrics.confusion_matrix(y_test, y_pred)
print(matrix)

# the area under the ROC curve 
fpr, tpr, thresholds = metrics.roc_curve(y_test,y_pred)
AUC = metrics.auc(fpr, tpr)
print("AUC : %0.3f " % AUC)

# Visualization

# ROC curve

plt.plot(fpr, tpr, color = 'g', label = 'ROC curve (area = %0.3f)' % AUC,)
plt.plot([0, 1], [0, 1],  color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.legend(loc="lower right")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Logistic regression : ROC Curve ')

# precison/recall curve
precision, recall, thresholds2 = metrics.precision_recall_curve(y_test, y_pred)
plt.plot(recall, precision, color='r')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Logistic regression : precision/recall Curve')


# performace measurement

# 20 pairs of samples in the form[I, P]
# "I" is the label for the sample, and "P" is the predicted probably that the label is 1
labels = np.array([[0, 0.18524473461553573],[1, 0.31228049571858607], [1, 0.7076439069418454], [1, 0.1520427923529021], [0, 0.102432503004548],	 
	[1, 0.39254869439324597], [1, 0.4130854279209383], [0, 0.22762431762294538], [1, 0.22191216235651612], [0, 0.5084389292748122], [0,	0.3081036182342043],
	[0, 0.3842660533792349], [1, 0.688010311941002], [0, 0.372089367663065], [1, 0.6572654078963468], [0, 0.20406096205822216], 
	[0,	 0.28054227612466304], [0, 0.09761098112381061], [0, 0.5383905522397453], [1, 0.5380759531966481]])

# task: set the threshold for predict label = 1 as 0.3, 0.5, 0.7, and choose the best threshold for classifier 


threshold_list = [0.3, 0.5, 0.7]

def	perf_stat(arr, thresh):
	y_true = arr[:,0]
	y_pred = [int(i >= thresh ) for i in arr[:,1]]
	confusion_matrix = metrics.confusion_matrix(y_true,	y_pred)
	F =	metrics. f1_score (y_true, y_pred)
	precision =	confusion_matrix[1,1] /	(confusion_matrix[1,1] + confusion_matrix[0,1])
	recall = confusion_matrix[1,1] / (confusion_matrix[1,1] + confusion_matrix[1,0])
	FPR	= confusion_matrix[0,1]	/ (confusion_matrix[0,1] + confusion_matrix[0,0])
	TPR	= confusion_matrix[1,1]	/ (confusion_matrix[1,1] + confusion_matrix[1,0])
	return [F, precision, recall, FPR, TPR]


#in	binary classification the count	of true	negatives is C{0,0}, false negatives is C{1,0}, true positives is C{1,1} and false positives is	C{0,1}

for thresh in threshold_list:
	result = perf_stat(labels, thresh)
	print ('When threshold is %s :' % thresh)
	print ('F1 score is %.4f' % result[0])
	print ('precision score is %.4f' % result[1])
	print ('recall score is %.4f' % result[2])
	print ('FPR	is %.4f' % result[3])
	print ('TPR	is %.4f' % result[4])

F = []
for thresh in threshold_list:
	F.append(perf_stat(labels, thresh)[0])
F_score = np.array(F)

# the best threshold for classifier based on the largest F score

print("the best threshold is %s" % threshold_list[F_score.argmax()])
















