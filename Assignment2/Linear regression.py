# Linear regression

import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

# get data

head = pd.read_csv("/Users/admin/Downloads/data/Afifi_book/LungSchema", header = None, sep=' ')
lung = pd.read_csv("/Users/admin/Downloads/data/Afifi_book/Lung.txt", sep=' ', header=None)
lung.columns = head[1].values
lung[["FEV1_father","Age_father","Height_father_inch"]][:7]
print(lung.head())

# do a scatter plot

y = lung["FEV1_father"]
x = lung[["Age_father"]]
plt.ylabel('FEV1_father')
plt.xlabel('Age_father')
plt.scatter(x,y);
plt.show()

# Doing a linear regression

x2 = sm.add_constant(x)
est_lung = sm.OLS(y,x2).fit()
est_lung.summary()
paramter = est_lung. params. values
print(paramter) # parameter for linear regression model

# putting it on the scatter plot


y_predict=x2.dot(est_lung.params.values)
plt.scatter(x,y)
plt.plot(x,y_predict,'r--')
plt.show()