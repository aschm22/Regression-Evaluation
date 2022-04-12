###################################
#####    Open ML Evaluation   #####
###################################

# data obtained:
# https://www.openml.org/d/41187
import sklearn as skl
from sklearn import datasets
from sklearn import metrics
from sklearn import model_selection
import pandas as pd
import numpy as np
import scipy as sp
from matplotlib import pyplot
from sklearn.ensemble import BaggingClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import learning_curve
from sklearn.svm import SVR
from sklearn.dummy import DummyRegressor
from sklearn.neighbors import KNeighborsRegressor

# load the data
volcano = datasets.fetch_openml(data_id=41187) 

# transform nominal values into numeric values
transform = ColumnTransformer([("encoder", OneHotEncoder(sparse=False), [5])], remainder="passthrough")
new_data = transform.fit_transform(volcano.data)
vol_new_data = pd.DataFrame(new_data, columns=transform.get_feature_names())

# linear regression rmse
lr = LinearRegression()
scores_lr = model_selection.cross_validate(lr, vol_new_data, volcano.target, cv=10, scoring="neg_root_mean_squared_error")
value_lr = scores_lr["test_score"]
rmse_lr = 0 - scores_lr["test_score"]
print("The linear regression average RMSE is:", round(rmse_lr.mean(),2))


# linear regression learning curve - blue line on graph
train_sizes_lr, train_scores_lr, test_scores_lr, fit_times_lr, score_times_lr = skl.model_selection.learning_curve(lr,
                                                                                                                   vol_new_data, 
                                                                                                                   volcano.target, 
                                                                                                                   train_sizes = ([0.2,0.4,0.6,0.8,1]),
                                                                                                                   cv=10, 
                                                                                                                   scoring="neg_root_mean_squared_error", 
                                                                                                                   return_times=True)
print("The training sizes for learning curve regression were:", train_sizes_lr)
time_lrcurve = fit_times_lr.sum() + score_times_lr.sum()
print("The total learning curve time for linear regression is:", round(time_lrcurve, 2), "seconds")
rmse_lrcurve = 0 - test_scores_lr
rmse_mean_lrcurve = rmse_lrcurve.mean(axis=1)
lr_plot = pyplot.plot(train_sizes_lr, rmse_mean_lrcurve)
lr_x_label = pyplot.xlabel("# of training instances")
lr_y_label = pyplot.ylabel("RMSE")



# Support Vector Machine regressor
svr = SVR()
scores_svr = model_selection.cross_validate(svr, vol_new_data, volcano.target, cv=10, scoring="neg_root_mean_squared_error")
rmse_svr = 0 - scores_svr["test_score"]
print("The SVM average RMSE is:", round(rmse_svr.mean(),3))

# Support vector machine learning curve - orange line on graph
train_sizes_svr, train_scores_svr, test_scores_svr, fit_times_svr, scores_times_svr = skl.model_selection.learning_curve(svr, 
                                                                                                                         vol_new_data, 
                                                                                                                         volcano.target, 
                                                                                                                         train_sizes = ([0.2,0.4,0.6,0.8,1]), 
                                                                                                                         cv=10, 
                                                                                                                         scoring="neg_root_mean_squared_error", 
                                                                                                                         return_times=True)
time_svrcurve = fit_times_svr.sum() + scores_times_svr.sum()
print("The total learning curve time for SVM regression is:", round(time_svrcurve, 2), "seconds")
rmse_svrcurve = 0 - test_scores_svr
rmse_mean_svrcurve = rmse_svrcurve.mean(axis=1)
svr_plot = pyplot.plot(train_sizes_svr, rmse_mean_svrcurve)


# bagging regression
br = BaggingRegressor()
scores_br = model_selection.cross_validate(br, vol_new_data, volcano.target, cv=10, scoring="neg_root_mean_squared_error")
rmse_br = 0 - scores_br["test_score"]
print("The bagging regression average RMSE is:", round(rmse_br.mean(), 2))

# bagging regression learning curve - green line
train_sizes_br, train_scores_br, test_scores_br, fit_times_br, scores_times_br = skl.model_selection.learning_curve(br,
                                                                                                                   vol_new_data,
                                                                                                                   volcano.target, 
                                                                                                                   train_sizes = ([0.2,0.4,0.6,0.8,1]),
                                                                                                                   cv=10, 
                                                                                                                   scoring="neg_root_mean_squared_error",
                                                                                                                   return_times=True)
time_brcurve = fit_times_br.sum() + scores_times_br.sum()
print("The total learning curve time for bagging regression is:", round(time_brcurve, 2), "seconds")
rmse_brcurve = 0 - test_scores_br
rmse_mean_brcurve = rmse_brcurve.mean(axis=1)
br_plot = pyplot.plot(train_sizes_br, rmse_mean_brcurve)


# dummy regression
dr = DummyRegressor()
scores_dr = model_selection.cross_validate(dr, vol_new_data, volcano.target, cv=10, scoring="neg_root_mean_squared_error")
rmse_dr = 0 - scores_dr["test_score"]
print("The dummy regression average RMSE is:", round(rmse_dr.mean(), 2))

# dummy regression learning curve - red line
train_sizes_dr, train_scores_dr, test_scores_dr, fit_times_dr, scores_times_dr = skl.model_selection.learning_curve(dr, 
                                                                                                                   vol_new_data, 
                                                                                                                   volcano.target, 
                                                                                                                   train_sizes = ([0.2,0.4,0.6,0.8,1]),
                                                                                                                   cv=10, 
                                                                                                                   scoring="neg_root_mean_squared_error",
                                                                                                                   return_times=True)
time_drcurve = fit_times_dr.sum() + scores_times_dr.sum()
print("The total learning curve time for dummy regression is:", round(time_drcurve, 2), "seconds")
rmse_drcurve = 0 - test_scores_dr
rmse_mean_drcurve = rmse_drcurve.mean(axis=1)
dr_plot = pyplot.plot(train_sizes_dr, rmse_mean_drcurve)



# Decision tree regression
dtr = DecisionTreeRegressor(min_samples_leaf = 2)
scores_dtr = model_selection.cross_validate(dtr, vol_new_data, volcano.target, scoring="neg_root_mean_squared_error")
rmse_dtr = 0 - scores_dtr["test_score"]
print("The decision tree regression average RMSE is:", round(rmse_dtr.mean(),2))

# learning curve decision tree regression - purple line
train_sizes_dtr, train_scores_dtr, test_scores_dtr, fit_times_dtr, scores_times_dtr = skl.model_selection.learning_curve(dtr,
                                                                                                                      vol_new_data, 
                                                                                                                      volcano.target, 
                                                                                                                      train_sizes = ([0.2,0.4,0.6,0.8,1]),
                                                                                                                      cv = 10,
                                                                                                                      scoring="neg_root_mean_squared_error",
                                                                                                                      return_times=True)
time_dtrcurve = fit_times_dtr.sum() + scores_times_dtr.sum()
print("The total learning curve time for dummy regression is:", round(time_dtrcurve, 2), "seconds")
rmse_dtrcurve = 0 - test_scores_dtr
rmse_mean_dtrcurve = rmse_dtrcurve.mean(axis=1)
dtr_plot = pyplot.plot(train_sizes_dtr, rmse_mean_dtrcurve)


# K-nearest neighbors regression
knn = KNeighborsRegressor(n_neighbors=2)
scores_knn = model_selection.cross_validate(knn, vol_new_data, volcano.target, scoring="neg_root_mean_squared_error")
rmse_knn = 0 - scores_knn["test_score"]
print("The k-nearest neighbors regression average RMSE is:", round(rmse_knn.mean(), 2))

# K-nearest neighbors learning curve
train_sizes_knn, train_scores_knn, test_scores_knn, fit_times_knn, scores_times_knn = skl.model_selection.learning_curve(dtr,
                                                                                                                      vol_new_data, 
                                                                                                                      volcano.target, 
                                                                                                                      train_sizes = ([0.2,0.4,0.6,0.8,1]),
                                                                                                                      cv = 10,
                                                                                                                      scoring="neg_root_mean_squared_error",
                                                                                                                      return_times=True)
time_knncurve = fit_times_knn.sum() + scores_times_knn.sum()
print("The total learning curve time for K-nearest neighbors is:", round(time_knncurve, 2), "seconds")
rmse_knncurve = 0 - test_scores_knn
rmse_mean_knncurve = rmse_knncurve.mean(axis=1)
knn_plot = pyplot.plot(train_sizes_dtr, rmse_mean_knncurve)



# now compare the computational and testing times for the last point of alll models learning curves
train_sizes_lr, train_scores_lr, test_scores_lr, fit_times_lr, score_times_lr = skl.model_selection.learning_curve(lr,
                                                                                                                   vol_new_data, 
                                                                                                                   volcano.target, 
                                                                                                                   train_sizes = ([1]),
                                                                                                                   cv=10, 
                                                                                                                   scoring="neg_root_mean_squared_error", 
                                                                                                                   return_times=True)
print("The training sizes for learning curve regression were:", train_sizes_lr)
time_lrcurve = fit_times_lr.sum() + score_times_lr.sum()
print("The total learning curve time for linear regression is:", round(time_lrcurve, 2), "seconds")

train_sizes_svr, train_scores_svr, test_scores_svr, fit_times_svr, scores_times_svr = skl.model_selection.learning_curve(svr, 
                                                                                                                         vol_new_data, 
                                                                                                                         volcano.target, 
                                                                                                                         train_sizes = ([1]), 
                                                                                                                         cv=10, 
                                                                                                                         scoring="neg_root_mean_squared_error", 
                                                                                                                         return_times=True)
time_svrcurve = fit_times_svr.sum() + scores_times_svr.sum()
print("The total learning curve time for SVM regression is:", round(time_svrcurve, 2), "seconds")

train_sizes_br, train_scores_br, test_scores_br, fit_times_br, scores_times_br = skl.model_selection.learning_curve(br,
                                                                                                                   vol_new_data,
                                                                                                                   volcano.target, 
                                                                                                                   train_sizes = ([1]),
                                                                                                                   cv=10, 
                                                                                                                   scoring="neg_root_mean_squared_error",
                                                                                                                   return_times=True)
time_brcurve = fit_times_br.sum() + scores_times_br.sum()
print("The total learning curve time for bagging regression is:", round(time_brcurve, 2), "seconds")

train_sizes_dr, train_scores_dr, test_scores_dr, fit_times_dr, scores_times_dr = skl.model_selection.learning_curve(dr, 
                                                                                                                   vol_new_data, 
                                                                                                                   volcano.target, 
                                                                                                                   train_sizes = ([1]),
                                                                                                                   cv=10, 
                                                                                                                   scoring="neg_root_mean_squared_error",
                                                                                                                   return_times=True)
time_drcurve = fit_times_dr.sum() + scores_times_dr.sum()
print("The total learning curve time for dummy regression is:", round(time_drcurve, 2), "seconds")

train_sizes_dtr, train_scores_dtr, test_scores_dtr, fit_times_dtr, scores_times_dtr = skl.model_selection.learning_curve(dtr,
                                                                                                                      vol_new_data, 
                                                                                                                      volcano.target, 
                                                                                                                      train_sizes = ([1]),
                                                                                                                      cv = 10,
                                                                                                                      scoring="neg_root_mean_squared_error",
                                                                                                                      return_times=True)
time_dtrcurve = fit_times_dtr.sum() + scores_times_dtr.sum()
print("The total learning curve time for decision tree regression is:", round(time_dtrcurve, 2), "seconds")

train_sizes_knn, train_scores_knn, test_scores_knn, fit_times_knn, scores_times_knn = skl.model_selection.learning_curve(dtr,
                                                                                                                      vol_new_data, 
                                                                                                                      volcano.target, 
                                                                                                                      train_sizes = ([1]),
                                                                                                                      cv = 10,
                                                                                                                      scoring="neg_root_mean_squared_error",
                                                                                                                      return_times=True)
time_knncurve = fit_times_knn.sum() + scores_times_knn.sum()
print("The total learning curve time for K-nearest neighbors is:", round(time_knncurve, 2), "seconds")


# now find statistical significance between bagging regression and other models
print(sp.stats.ttest_rel(rmse_lr, rmse_br))
print(sp.stats.ttest_rel(rmse_dr, rmse_br))
print(sp.stats.ttest_rel(rmse_svr, rmse_br))
