import pandas as pd
import numpy as np
import csv as csv
import operator
import matplotlib.pyplot as plt

from datetime import date
from mpl_toolkits.mplot3d import Axes3D

from sklearn import linear_model
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor

##############
from sklearn import cross_validation
##############
def rmse(reslt,predt,tst):
	return (np.sqrt(((sum((reslt - predt)**2)))/len(tst)))

#############################################################################################
#										Data Reading Section								#
#############################################################################################

#Data taken from UCI Repository and divided into train.csv and test.csv

#train data - train.csv file
train_data = pd.read_csv("Data_Set/train.csv", header=0)

#test data - test.csv file
test_data = pd.read_csv("Data_Set/test.csv", header=0)

#############################################################################################
#										Data Cleaning Section								#
#############################################################################################

print 'Data cleaning'

#Cleaning training data
#Storing target values to train_count variable
train_count = train_data['count'].values

t_data = train_data.values
dy_hr = [[0 for x in xrange(24)] for x in xrange(7)]
print dy_hr

for i in t_data:
	dy_hr[i[7]][i[5]]+=i[16]
print dy_hr


#dropping features in train_data that's not needed
train_data = train_data.drop(['instant','date','casual','registered','count'], axis=1)
train = train_data.values

#Cleaning test data
dates = test_data['date'].values
instant = test_data['instant'].values
res = test_data['count'].values

#dropping features in train_data that's not needed
test_data = test_data.drop(['instant','date','casual','registered','count'], axis=1)
test = test_data.values

######################
# cross validating
#####################
X_train, X_test, y_train, y_test = cross_validation.train_test_split(train, train_count, test_size=0.4, random_state=0)

#############################################################################################
#										Training Section									#
#############################################################################################

print 'Training'

#Support Vector Regression
rbf = SVR(kernel = 'rbf', C=1e3, gamma = 0.1)
#rbf.fit(train, train_count)
rbf.fit(X_train, y_train)
print 'svr with rbf ',rbf.score(X_test, y_test)

#Bayesian Ridge Regression
clf = linear_model.BayesianRidge(compute_score=True)
#clf.fit(train,train_count)
clf.fit(X_train, y_train)
print 'Bayesian Ridge Regression ',clf.score(X_test, y_test)

#Linear Regression
ols = linear_model.LinearRegression()
#ols.fit(train,train_count)
ols.fit(X_train, y_train)
print 'Linear Regressor  ',ols.score(X_test, y_test)


	
#Gradient Boosting Regression
gradient = GradientBoostingRegressor(n_estimators = 2500, max_depth = 4, learning_rate = 0.00001, random_state = 0, loss = 'huber')
#gradient.fit(train,train_count)
gradient.fit(X_train, y_train)
print 'Gradient Boosting Regression ',gradient.score(X_test, y_test)


#Random Forest Regression
forest = RandomForestRegressor(n_estimators=251)
#forest.fit(train,train_count)
forest.fit(X_train, y_train)
print 'Random Forest Regression Accuracy  ',forest.score(X_test, y_test)
#############################################################################################
#										Predicting Section									#
#############################################################################################

print 'Predicting'

#Support Vector Regressor prediction
pred_rbf = rbf.predict(test).astype(int)

#Bayesian Ridge Regressor prediction
pred_clf = clf.predict(test).astype(int)

#Linear Regressor prediction
pred_ols = ols.predict(test).astype(int)

#Gradient Boosting Regressor prediction
pred_gradient = gradient.predict(test).astype(int)

#Random Forest Regressor prediction
pred_forest = forest.predict(test).astype(int)

#############################################################################################
#										Root Mean Square Error Section						#
#############################################################################################

#Support Vector Regressor rmse
rmse_rbf = rmse(res, pred_rbf, test)

#Bayesian Ridge Regressor rmse
rmse_clf = rmse(res, pred_clf, test)

#Linear Regressor rmse
rmse_ols = rmse(res, pred_ols, test)

#Gradient Boosting rmse
rmse_forest = rmse(res, pred_forest, test)

#Random Forest Regressor rmse
rmse_gradient = rmse(res,pred_gradient, test)

#############################################################################################
#										Graph Plotting Section								#
#############################################################################################

#model - rmse error data dictionary
model_rmse = {'Bayesian Regression':rmse_clf,'Linear Regression':rmse_ols, 'Random Forest Regressor':rmse_forest, 
			'Gradient Boosting Regressor':rmse_gradient, 'SVM Radial Basis':rmse_rbf}

#sorting the models in acsending order based on rmse
sorted_model_rmse = sorted(model_rmse.items(), key = operator.itemgetter(1))

#extracting models and errors from the dictionary
modls=[]
err=[]
for i in sorted_model_rmse:
	modls.append(i[0])
	err.append(i[1])

fig = plt.figure()
ax1 = fig.add_subplot(111, projection = '3d')

#plotting model vs rmse
plt.figure(figsize = (6, 5))
plt.title("Models vs RMSE")
models = np.array([1, 2, 3, 4, 5])
plt.xticks(models,modls)
plt.plot(models,err)
plt.xlabel("Models")
plt.ylabel("Error")
plt.show()

#############################################################################################
#										Data Storage Section								#
#############################################################################################

print 'Writing file'

#Bayesian Ridge result file
pred_file_clf = open("results/results_bayesian_ridge.csv","wb")
open_csv_clf = csv.writer(pred_file_clf)
open_csv_clf.writerow(["instant", "date", "count"])
open_csv_clf.writerows(zip(instant, dates, pred_clf))
pred_file_clf.close()

#Linear Regression results file
pred_file_ols = open("results/results_linear_regression.csv","wb")
open_csv_ols = csv.writer(pred_file_ols)
open_csv_ols.writerow(["instant", "date", "count"])
open_csv_ols.writerows(zip(instant, dates, pred_ols))
pred_file_ols.close()

#Radial Basis SVM Regression result file
pred_file_rbf = open("results/results_radial_basis.csv","wb")
open_csv_rbf = csv.writer(pred_file_rbf)
open_csv_rbf.writerow(["instant", "date", "count"])
open_csv_rbf.writerows(zip(instant, dates, pred_rbf))
pred_file_rbf.close()

#Random Forest Regression result file
pred_file_forest = open("results/results_random_forest.csv","wb")
open_csv_forest = csv.writer(pred_file_forest)
open_csv_forest.writerow(["instant", "date", "count"])
open_csv_forest.writerows(zip(instant, dates, pred_forest))
pred_file_forest.close()

#Gradient Boosting Regression file
pred_file_gradient = open("results/results_gradient_boost.csv","wb")
open_csv_gradient = csv.writer(pred_file_gradient)
open_csv_gradient.writerow(["instant", "date", "count"])
open_csv_gradient.writerows(zip(instant, dates, pred_gradient))
pred_file_gradient.close()

print 'Done'
